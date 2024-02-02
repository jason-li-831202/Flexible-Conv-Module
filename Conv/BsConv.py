import torch,  math
from torch import nn
from typing import Union, Tuple
from collections import OrderedDict


__all__ = ["BlueprintSeparableUConv2d", "BlueprintSeparableSConv2d"]
# Blueprint Separable Residual Network for Efficient Image Super-Resolution (https://arxiv.org/abs/2205.05996)
# CVPR 2022

class BlueprintSeparableUConv2d(nn.Module):
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]], 
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 0, 
                        dilation: int = 1, 
                        bias: bool = True, 
                        padding_mode: str = 'zeros',
                        use_bn: bool = False,
                        bn_kwargs: dict = None):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        self.pw_part = nn.Sequential(OrderedDict([
            ('pw_conv', nn.Conv2d(  in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)),
            ('pw_bn', torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs) if use_bn else nn.Identity() ),
        ]))


        self.dw_part = nn.Sequential(OrderedDict([
            ('dw_conv', nn.Conv2d(  in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=out_channels,
                                    bias=bias,
                                    padding_mode=padding_mode))
        ]))

    def forward(self, x):
        return self.dw_part(self.pw_part(x))


class BlueprintSeparableSConv2d(nn.Module):
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]], 
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 0, 
                        dilation: int = 1, 
                        bias: bool = True, 
                        padding_mode: str = 'zeros',
                        ratio: float = 0.25, 
                        min_internal_channels: int = 4,
                        use_bn: bool = False,
                        bn_kwargs: dict = None):
        super().__init__()

        # check arguments
        assert 0.0 <= ratio <= 1.0
        mid_channels = min(in_channels, max(min_internal_channels, math.ceil(ratio * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1_part = nn.Sequential(OrderedDict([
            ('pw1_conv', nn.Conv2d( in_channels=in_channels,
                                    out_channels=mid_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)),
            ('pw1_bn', torch.nn.BatchNorm2d(num_features=mid_channels, **bn_kwargs) if use_bn else nn.Identity() ),
        ]))

        # pointwise 2
        self.pw2_part = nn.Sequential(OrderedDict([
            ('pw2_conv', nn.Conv2d( in_channels=mid_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)),
            ('pw2_bn', torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs) if use_bn else nn.Identity() ),
        ]))

        # depthwise
        self.dw_part = nn.Sequential(OrderedDict([
            ('dw_conv', nn.Conv2d(  in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=out_channels,
                                    bias=bias,
                                    padding_mode=padding_mode))
        ]))
    
    def forward(self, x):
        return self.dw_part(self.pw2_part(self.pw1_part(x)))
    
    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")
