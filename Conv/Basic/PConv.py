import torch, math
import numpy as np
from torch import mean, nn
from torch.nn import functional as F
from typing import Union, Tuple, Optional

# Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks (https://arxiv.org/abs/2303.03667)
# CVPR 2023

# Partial Convolutional 
class PConv2d(nn.Module):

    def __init__(self,  in_channels: int,  
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]] = 3,
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 1, 
                        dilation: int = 1, 
                        groups: int = 1, 
                        n_div: int = 4,
                        bias: bool = False,
                        padding_mode: str = 'zeros'):
        super().__init__()
        self.partial_channels = in_channels // n_div
        self.untouched_channels = in_channels - self.partial_channels

        self.partial_conv = nn.Conv2d(self.partial_channels, self.partial_channels, kernel_size=kernel_size, 
                                       stride=stride, padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode, bias=bias)
        self.concat_conv = nn.Conv2d(self.partial_channels+self.untouched_channels, out_channels, kernel_size = 1)

    def forward_split_cat(self, x):
        part1_out, part2_out = torch.split(x, [self.partial_channels, self.untouched_channels], dim=1)
        part1_out = self.partial_conv(part1_out)
        out = torch.cat((part1_out, part2_out), 1)
        return self.concat_conv(out)