import torch
import warnings
from torch import nn

from Conv.Basic.common import ConvBN

__all__ = ["SPPF", "SPPFBlock", "SimSPPFBlock"]

# Spatial Pyramid Pooling-Fast
class SPPF(nn.Module):
    def __init__(self, pool_size: int = 5, mode: str = "max"):
        super().__init__()
        
        if mode =='max':
            self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        else :
            self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        o1 = self.pool(x)
        o2 = self.pool(o1)
        o3 = self.pool(o2)
        return torch.cat([x, o1, o2, o3], dim=1)
    

class SPPFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: 5, mode: str = "max"):
        super().__init__()

        hidden_channels = in_channels // 2  # hidden channels
        self.pw1_conv = ConvBN(in_channels, hidden_channels, kernel_size=1, stride=1, 
                                activation=nn.SiLU())
        self.pw2_conv = ConvBN(hidden_channels * 4, out_channels, kernel_size=1, stride=1,
                                activation=nn.SiLU())
        
        self.sppf = SPPF(pool_size=pool_size, mode=mode)

    def forward(self, x):
        x = self.pw1_conv(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.pw2_conv(self.sppf(x))
        

class SimSPPFBlock(nn.Module):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels: int, out_channels: int, pool_size: 5, mode: str = "max"):
        super().__init__()
        hidden_channels = in_channels // 2  # hidden channels
        self.pw1_conv = ConvBN(in_channels, hidden_channels, kernel_size=1, stride=1, 
                                activation=nn.ReLU(inplace=True))
        self.pw2_conv = ConvBN(hidden_channels * 4, out_channels, kernel_size=1, stride=1,
                                activation=nn.ReLU(inplace=True))
        self.sppf = SPPF(pool_size=pool_size, mode=mode)
 
    def forward(self, x):
        x = self.pw1_conv(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.pw2_conv(self.sppf(x))
