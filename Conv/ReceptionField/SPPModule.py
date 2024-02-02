import torch
import warnings
from torch import nn

from Conv.Basic.common import ConvBN

__all__ = ["SPP", "SPPBlock"]
# Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition (https://arxiv.org/abs/1406.4729)
# CVPR 2015

# Spatial Pyramid Pooling
class SPP(nn.Module):
    def __init__(self, pool_sizes: list = [5, 9, 13], mode: str = "max"):
        super(SPP, self).__init__()

        multi_pools = []
        if mode =='max':
            multi_pools = [nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2) \
                            for pool_size in pool_sizes]
        else :
            multi_pools = [nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2) \
                            for pool_size in pool_sizes]
        self.spp_block = nn.ModuleList(multi_pools)
        self.branch_num = len(pool_sizes) + 1

    def forward(self, x):
        features = [pool(x) for pool in self.spp_block]
        features = torch.cat(features + [x], dim=1)

        return features
    

class SPPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: list = [5, 9, 13], mode: str = "max"):
        super().__init__()

        hidden_channels = in_channels // 2  # hidden channels
        self.pw1_conv = ConvBN(in_channels, hidden_channels, kernel_size=1, stride=1, 
                                activation=nn.ReLU(inplace=True))
        self.spp = SPP(pool_sizes=pool_sizes, mode=mode)
        self.pw2_conv = ConvBN(hidden_channels * self.spp.branch_num, out_channels, kernel_size=1, stride=1,
                                activation=nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.pw1_conv(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.pw2_conv(self.spp(x))
