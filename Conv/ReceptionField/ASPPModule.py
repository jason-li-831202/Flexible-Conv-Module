import torch, math
import warnings
from torch import nn
from torch.nn import functional as F

from Conv.Basic.common import ConvBN

__all__ = ["ASPPv2", "ASPPv3", "ASPPv3Block"]
# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs (https://arxiv.org/abs/1606.00915v2)
# CVPR 2017

# Rethinking Atrous Convolution for Semantic Image Segmentation (https://arxiv.org/abs/1706.05587)
# CVPR 2017

# Atrous Spatial Pyramid Pooling
class ASPPv2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size: int = 3, dilations: list = [6, 12, 18, 24]):
        super(ASPPv2, self).__init__()

        multi_atrous = []
        for dilation in dilations: 
            padding = dilation*(kernel_size-1)//2
            multi_atrous.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, 
                                          padding=padding, dilation=dilation, bias=True))
        self.aspp_block = nn.ModuleList(multi_atrous)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = [atrous(x) for atrous in self.aspp_block]
        return torch.cat(out, dim=1)
    
class _ImagePooling(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.pw1_conv = ConvBN(in_channel, out_channel, kernel_size=1, stride=1, 
                                activation=nn.ReLU(inplace=True))
    def forward(self, x):
        _, _, w, h = x.shape
        x = self.global_avg_pool(x) # (B, in_c, 1, 1)
        x = self.pw1_conv(x) # (B, out_c, 1, 1)
        x = F.interpolate(x, size=(w, h), mode="bilinear", align_corners=False) # (B, out_c, h, w)
        return x
    
class ASPPv3(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, dilations: list = [12, 24, 36]):
        super(ASPPv3, self).__init__()
        
        multi_atrous = [ConvBN(in_channel, out_channel, kernel_size=1, stride=1, 
                                bias=False, activation=nn.ReLU(inplace=True))]
        for dilation in dilations: 
            padding = dilation*(kernel_size-1)//2
            multi_atrous.append(ConvBN(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, 
                                       dilation=dilation, bias=False, activation=nn.ReLU(inplace=True)))
        self.aspp_block = nn.ModuleList(multi_atrous)
        self.branch_num = len(dilations) + 1
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = [atrous(x) for atrous in self.aspp_block]
        return torch.cat(out, dim=1)
    
class ASPPv3Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilations: list = [12, 24, 36]):
        super().__init__()
        self.image_pooling = _ImagePooling(in_channel=in_channels, out_channel=out_channels)
        self.aspp = ASPPv3(in_channel=in_channels, out_channel=out_channels, kernel_size=kernel_size, dilations=dilations)
        self.pw_conv = ConvBN(out_channels * (self.aspp.branch_num + 1), out_channels, kernel_size=1, stride=1,
                                activation=nn.ReLU(inplace=True))
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            image_features = self.image_pooling(x)
            return self.pw_conv(torch.cat([image_features, self.aspp(x)], dim=1) )
