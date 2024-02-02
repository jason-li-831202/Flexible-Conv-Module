import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple, Optional

# PSConv: Squeezing Feature Pyramid into One Compact Poly-Scale Convolutional Layer (https://arxiv.org/abs/2007.06191)
# ECCV 2020

class PSConv2d(nn.Module):
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]] = 3, 
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 1, 
                        dilation: int = 1, 
                        groups: int = 1, 
                        parts: int = 4, 
                        bias: bool = False,
                        padding_mode: str = 'zeros'):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=groups * parts, padding_mode=padding_mode, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=groups * parts, padding_mode=padding_mode, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, padding_mode=padding_mode, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // (groups * parts)
        _out_channels = out_channels // (groups * parts)
        for i in range(parts):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
                self.mask[((i + parts // 2) % parts + j * groups) * _out_channels: ((i + parts // 2) % parts + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.gwconv_shift(x_merge)
        return self.gwconv(x) + self.conv(x) + x_shift