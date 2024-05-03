import torch
from sample_factory.model.encoder import Encoder, ResBlock
from sample_factory.model.model_utils import  nonlinearity
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.utils import log
from torch import nn as nn
from typing import Tuple
from learning.epom_config import EncoderConfig


class CustomEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # noinspection Pydantic
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder_config)
        self._length = obs_space['obs'].shape[0]
        input_ch = obs_space['obs'].shape[1]

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
                layers.append(ResBlock(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        layers.append(nonlinearity(cfg, inplace = True))

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
        if self.encoder_cfg.with_direction:
            coordinates_x = torch.cat([x['xy'], x['target_xy'], x['direction']], -1)
        else:
            coordinates_x = torch.cat([x['xy'], x['target_xy']], -1)
        coordinates_scale = 64.0
        abs_coordinates = torch.max(torch.abs(coordinates_x), torch.tensor(coordinates_scale)) # abs绝对值
        coordinates_x /= abs_coordinates
        coordinates_x = self.coordinates_mlp(coordinates_x) # linear层默认只处理最后一个维度in_features

        x = x['obs']
        batch_size, k, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        x = self.conv_head(x)
        x = x.contiguous().view(batch_size, k, -1)
        x = torch.cat([x, coordinates_x], -1)

        x = x.view(-1, self.encoder_out_size)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)
        return x

