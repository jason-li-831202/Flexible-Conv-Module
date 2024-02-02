import torch
import torch.nn as nn
import torchvision

'''
FLOPs  = 356.5 M
params = 5.6 K
'''

__all__ = ["LKAConv", "deformable_LKAConv", "LKABlock"]
# Beyond Self-Attention: Deformable Large Kernel Attention for Medical Image Segmentation (https://arxiv.org/abs/2309.00121)
# CVPR 2023

class LKAConv(nn.Module):
    def __init__(self,  channels: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw_spatial_conv = nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=9, groups=channels, dilation=3)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)


    def forward(self, x):
        u = x.clone()        
        attn = self.dw_conv(x)
        attn = self.dw_spatial_conv(attn)
        attn = self.pw_conv(attn)

        return u * attn

class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out

class deformable_LKAConv(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = DeformConv(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw_spatial_conv = DeformConv(channels, channels, kernel_size=7, stride=1, padding=9, groups=channels, dilation=3)
        self.pw_conv = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.dw_conv(x)
        attn = self.dw_spatial_conv(attn)
        attn = self.pw_conv(attn)

        return u * attn
    
# (Deformable) Large Kernel Attention
class LKABlock(nn.Module):
    def __init__(self, channels: int, deformable_mode: bool = False):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            deformable_mode: use deformable conv.
        """
        super(LKABlock, self).__init__()

        self.pw1_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKAConv(channels) if not deformable_mode else deformable_LKAConv(channels)
        self.pw2_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.pw1_conv(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.pw2_conv(x)
        return x + shorcut

