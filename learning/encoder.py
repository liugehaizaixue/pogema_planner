from learning.epom_config import EncoderConfig
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.utils import log
from learning.resblock import ResBlock
from torch import nn
import torch


class ConvModule(nn.Module):
    """
    通用卷积模块，用于处理指定通道的输入
    """
    def __init__(self, encoder_cfg, input_channels, use_attention):
        super(ConvModule, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.use_attention = use_attention

        curr_input_channels = input_channels
        layers = []
        resnet_conf = [[encoder_cfg.num_filters, encoder_cfg.num_res_blocks]]
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.append(
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            for j in range(res_blocks):
                layers.append(ResBlock(encoder_cfg, out_channels, out_channels, use_attention))
            
            curr_input_channels = out_channels
        
        layers.append(nonlinearity(self.encoder_cfg, inplace = True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class MultiHeadConv(nn.Module):
    """
    整合两个卷积头的网络
    """
    def __init__(self, encoder_cfg, input_ch):
        super(MultiHeadConv, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.use_attention = encoder_cfg.use_attention
        self.multi_head = encoder_cfg.multi_head # encoder_cfg.multi_head # ? True or False

        if self.multi_head:
            # 前3通道卷积头
            self.conv_head_3ch = ConvModule(encoder_cfg, input_channels=3, use_attention=self.use_attention)
            # 后2通道卷积头
            self.conv_head_2ch = ConvModule(encoder_cfg, input_channels=input_ch-3, use_attention=self.use_attention)
        else:
            self.conv_head = ConvModule(encoder_cfg, input_channels=input_ch, use_attention=self.use_attention)

    def forward(self, obs):
        if self.multi_head:
            # 分离前3通道和后2通道
            obs_3ch = obs[:, :3, :, :]
            obs_2ch = obs[:, 3:, :, :]

            # 分别通过两个卷积头
            out_3ch = self.conv_head_3ch(obs_3ch)
            out_2ch = self.conv_head_2ch(obs_2ch)

            # 合并两个输出
            out_3ch = out_3ch.contiguous().view(out_3ch.size(0), -1)  # 展平为(batch_size, -1)
            out_2ch = out_2ch.contiguous().view(out_2ch.size(0), -1)  # 展平为(batch_size, -1)
            return torch.cat([out_3ch, out_2ch], dim=-1)
        else:
            return self.conv_head(obs)



class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # noinspection Pydantic
        print(cfg.encoder_config)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder_config)
        input_ch = obs_space['obs'].shape[0]
        log.debug('Num input channels: %d', input_ch)
        
        self.conv_head = MultiHeadConv(self.encoder_cfg, input_ch)

        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)

        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        if self.encoder_cfg.with_direction:
            coordinates_input_size = 6
        else:
            coordinates_input_size = 4
        self.coordinates_mlp = nn.Sequential(
            nn.Linear(coordinates_input_size, self.encoder_cfg.hidden_size),
            nonlinearity(cfg),
            nn.Linear(self.encoder_cfg.hidden_size, self.encoder_cfg.hidden_size),
            nonlinearity(cfg),
        )

        self.encoder_out_size = self.conv_head_out_size + self.encoder_cfg.hidden_size

        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                nonlinearity(self.encoder_cfg, inplace=True),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        coordinates_x = torch.cat([x['xy'], x['target_xy']], -1)
        coordinates_scale = 64.0
        abs_coordinates = torch.max(torch.abs(coordinates_x), torch.tensor(coordinates_scale)) # abs绝对值
        coordinates_x /= abs_coordinates
        if self.encoder_cfg.with_direction:
            coordinates_x = torch.cat([coordinates_x, x['direction']], -1)
        coordinates_x = self.coordinates_mlp(coordinates_x)

        obs = x['obs']
        # 归一化前两个通道
        obs[:, 0:2, :, :] = (obs[:, 0:2, :, :] + 1) / 2

        x = obs
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = torch.cat([x, coordinates_x], -1)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)
            
        return x

