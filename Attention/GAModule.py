import torch, math
from torch import nn
from torch.nn import init

try :
    from Conv.Basic.common import ConvBN
except :
    import sys
    sys.path.append("..")
    from Conv.Basic.common import ConvBN

# Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions (https://arxiv.org/abs/2112.05561)
# CVPR 2021
__all__ = ["GAMBlock"]

# Global_Attention_Mechanism
class GAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ratio: int = 4):
        super(GAMBlock, self).__init__()
 
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / ratio), in_channels)
        )
 
        self.spatial_attention = nn.Sequential(
            ConvBN(in_channels, int(in_channels / ratio), kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            ConvBN(int(in_channels / ratio), out_channels, kernel_size=7, padding=3, bias=True),
        )
 
    def forward(self, x):
        b, c, h, w = x.size()
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)

        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
 
        return out
