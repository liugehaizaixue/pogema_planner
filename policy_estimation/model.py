from argparse import Namespace

import gymnasium as gym
import torch
from gymnasium.spaces import Box
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.timing import Timing
from torch import nn

from learning.encoder import ResnetEncoder
from learning.epom_config import Experiment , EncoderConfig


class PolicyEstimationModel(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            print("cfg is None")
            exp = Experiment()
            cfg = Namespace(**exp.dict())

        full_size = 5 * 2 + 1
        observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(-1.0, 1, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            target_xy=Box(low=-1024, high=1024, shape=(2,), dtype=int),
            direction=Box(low=-1, high=1, shape=(2,), dtype=int),
        )
        self.encoder = ResnetEncoder(cfg, observation_space)
        encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder_config)
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.get_out_size(), 512),
            nonlinearity(encoder_cfg),
            nn.Linear(512, 512),
            nonlinearity(encoder_cfg),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.value_head(x)
        x = torch.squeeze(x, 1)
        return x
