import torch
from torch import nn
from learning.epom_config import CoreConfig
from learning.epom_config import Environment
from sample_factory.model.core import ModelCore
from typing import Tuple
import torch.nn.functional as F

class CustomCore(ModelCore):
    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)

        self.core_cfg: CoreConfig = CoreConfig(**cfg.core_config)
        self.env_config: Environment = Environment(**cfg.environment)
        self.core_output_size = input_size
        self.input_size = input_size
        self.memory_length = self.env_config.memory_length

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead= self.core_cfg.nhead, batch_first=True)
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=self.core_cfg.num_layers)



    def get_out_size(self) -> int:
        return self.core_output_size
        
    def forward(self, head_output, fake_rnn_states):

        head_output = head_output.view(-1, self.memory_length, self.input_size)
        x = self.core(head_output)
        
        # pooling
        # 定义权重参数
        weights = torch.arange(1, self.memory_length+1, dtype=torch.float)  # 从1到5的递增序列
        # 将权重参数转换为与输入张量的形状相匹配的张量
        weights = weights.unsqueeze(0).unsqueeze(-1)
        # 对权重参数进行归一化，确保权重之和为1
        weights = F.softmax(weights, dim=1)
        # 将权重参数广播到与输入张量相同的形状
        weights = weights.expand(x.size(0), -1, -1)
        x = x * weights
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), output_size=1).squeeze(dim=-1)

        return x, fake_rnn_states