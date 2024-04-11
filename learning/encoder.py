import torch
from sample_factory.model.encoder import Encoder, ResBlock
from sample_factory.model.model_utils import  nonlinearity
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.utils.utils import log
from torch import nn as nn

from learning.epom_config import ExperimentSettings


class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # noinspection Pydantic
        settings: ExperimentSettings = ExperimentSettings(**cfg.experiment_settings)
        input_ch = obs_space['obs'].shape[0]
        log.debug('Num input channels: %d', input_ch)

        resnet_conf = [[settings.pogema_encoder_num_filters, settings.pogema_encoder_num_res_blocks]]

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

        layers.append(nonlinearity(cfg))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

        self.coordinates_mlp = nn.Sequential(
            nn.Linear(4, settings.hidden_size),
            nn.ReLU(),
            nn.Linear(settings.hidden_size, settings.hidden_size),
            nn.ReLU(),
        )

        self.init_fc_blocks(self.conv_head_out_size + settings.hidden_size)

    def forward(self, x):

        coordinates_x = torch.cat([x['xy'], x['target_xy']], -1)
        coordinates_scale = 64.0
        abs_coordinates = torch.max(torch.abs(coordinates_x), torch.tensor(coordinates_scale)) # abs绝对值
        coordinates_x /= abs_coordinates
        coordinates_x = self.coordinates_mlp(coordinates_x)

        x = x['obs']
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = torch.cat([x, coordinates_x], -1)
        x = self.forward_fc_blocks(x)
        return x

