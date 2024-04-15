import torch
from torch import nn

from sample_factory.model.model_utils import ModelCore

class CustomCoreTransformer(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.core_output_size = cfg.rnn_size

        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.rnn_size, nhead= 8, batch_first=True)
        self.core = nn.TransformerEncoder(encoder_layer, num_layers=6)


    def forward(self, head_output, fake_rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        x = self.core(head_output)
        if not is_seq:
            x = x.squeeze(0)

        return x, fake_rnn_states