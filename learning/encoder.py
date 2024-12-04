from learning.epom_config import EncoderConfig
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.utils import log
from learning.resblock import ResBlock
from torch import nn
import torch




class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # noinspection Pydantic
        print(cfg.encoder_config)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder_config)
        self.use_attention = self.encoder_cfg.use_attention
        input_ch = obs_space['obs'].shape[0]
        log.debug('Num input channels: %d', input_ch)

        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            # noinspection PyTypeChecker
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                if j == 0:
                    layers.append(ResBlock(self.encoder_cfg, out_channels, out_channels, self.use_attention))
                else:
                    layers.append(ResBlock(self.encoder_cfg, out_channels, out_channels, None))

            curr_input_channels = out_channels

        layers.append(nonlinearity(self.encoder_cfg, inplace = True))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        if self.encoder_cfg.with_direction:
            coordinates_input_size = 6
        else:
            coordinates_input_size = 4
        self.coordinates_mlp = nn.Sequential(
            nn.Linear(coordinates_input_size, self.encoder_cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(self.encoder_cfg.hidden_size, self.encoder_cfg.hidden_size),
            nn.ReLU(),
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

