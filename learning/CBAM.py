import torch
from torch import nn

class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        x = x * channel_attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention
        return x


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = CBAMBlock(in_channels=512,reduction=4)
    output=cbam(input)

    