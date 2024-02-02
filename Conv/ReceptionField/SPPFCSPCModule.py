import torch
import warnings
from torch import nn
from collections import OrderedDict

from Conv.Basic.common import ConvBN
from Conv.ReceptionField.SPPFModule import SPPF

# YOLOv6 v3.0: A Full-Scale Reloading (https://arxiv.org/abs/2301.05586)
# CVPR 2023
__all__ = ["SPPFCSPCBlock"]

class SPPFCSPCBlock(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size: int = 3, groups: int = 1, ratio: float = 0.5, 
                 pool_size: int = 5, mode: str = "max"):
        """
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the conv kernel
            groups: Number of blocked connections from input channels to output channels.
            ratio: the ratio for internal_channels.
            pool_size: pool size for SPPF.
            mdoe: pool type for spp.
        """
        super(SPPFCSPCBlock, self).__init__()
        internal_channels  = int(2 * out_channels * ratio)  # hidden channels
        padding = (kernel_size-1)//2

        self.part1_branch = nn.Sequential(OrderedDict([
            ('pre_conv1x1-1', ConvBN(in_channels, internal_channels, kernel_size=1, stride=1, 
                                     groups=groups, activation=nn.SiLU())),
            ('pre_conv3x3', ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                     groups=groups, activation=nn.SiLU())),
            ('pre_conv1x1-2', ConvBN(internal_channels, internal_channels, kernel_size=1, stride=1, 
                                     groups=groups, activation=nn.SiLU())),
            ('spp', SPPF(pool_size=pool_size, mode=mode)),
            ('post_conv1x1', ConvBN(internal_channels*4, internal_channels, kernel_size=1, stride=1, 
                                     groups=groups, activation=nn.SiLU())),
            ('post_conv3x3', ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding,
                                     groups=groups, activation=nn.SiLU()))
        ]))

        self.part2_branch = ConvBN(in_channels, internal_channels, kernel_size=1, stride=1, 
                                    groups=groups, activation=nn.SiLU())

        self.pw_conv = ConvBN(2*internal_channels , out_channels, kernel_size=1, stride=1, 
                              groups=groups, activation=nn.SiLU())

    def forward(self, x):
        y1 = self.part1_branch(x)
        y2 = self.part2_branch(x)
        return self.pw_conv(torch.cat((y1, y2), dim=1))