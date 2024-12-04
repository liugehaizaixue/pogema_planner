from sample_factory.model.model_utils import nonlinearity
from learning.CBAM import CBAMBlock
from learning.SE import SEBlock
from torch import Tensor, nn
from learning.epom_config import EncoderConfig

class ResBlock(nn.Module):
    def __init__(self, cfg:EncoderConfig, input_ch, output_ch, use_attention=None):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]
        if use_attention == "cbam":
            # layers.append(nonlinearity(cfg)) # add non-linearity?
            layers.append(CBAMBlock(channel=output_ch, reduction=4))
        elif use_attention == "se":
            # layers.append(nonlinearity(cfg)) # add non-linearity?
            layers.append(SEBlock(output_ch, reduction=4))

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out