import torch
from torch import nn
from learning.epom_config import CoreConfig
from sample_factory.model.core import ModelCore
from typing import Tuple

class CustomCore(ModelCore):
    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)

        self.core_cfg: CoreConfig = CoreConfig(**cfg.core_config)
        self.core_output_size = input_size
        self.input_size = input_size
        self.memory_length = self.core_cfg.memory_length

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead= self.core_cfg.nhead, batch_first=True)
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=self.core_cfg.num_layers)



    def get_out_size(self) -> int:
        return self.core_output_size
        
    def forward(self, head_output, fake_rnn_states):

        head_output = head_output.view(-1, self.memory_length, self.input_size)
        x = self.core(head_output)

        return x, fake_rnn_states