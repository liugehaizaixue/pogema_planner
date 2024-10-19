from sample_factory.model.model_utils import nonlinearity
from learning.CBAM import CBAMBlock
from torch import Tensor, nn


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch, use_cbam=False):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]
        if use_cbam:
            nonlinearity(cfg), # add non-linearity?
            layers.append(CBAMBlock(channel=output_ch, reduction=4, kernel_size=3))

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out