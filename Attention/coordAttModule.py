import torch
import torch.nn as nn
from torch.nn import init
try :
    from Conv.Basic.common import ConvBN
except :
    import sys
    sys.path.append("..")
    from Conv.Basic.common import ConvBN

# Coordinate Attention for Efficient Mobile Network Design (https://arxiv.org/abs/2103.02907)
# CVPR 2021
__all__ = ["CoordAttBlock"]
# Coordinate_Attention_Module

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttBlock(nn.Module):
    def __init__(self, channels: int, internal_reduce: int = 32, min_internal_dims: int = 8):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            internal_reduce: the ratio for internal_channels.
            min_internal_dims: the minimum dim of the internal_channels, default 8.
        """
        super(CoordAttBlock, self).__init__()

        internal_channels = max(channels // internal_reduce, min_internal_dims)

        # Coordinate Information Embedding
        self.h_avg_pooling = nn.AdaptiveAvgPool2d((None, 1))
        self.w_avg_pooling = nn.AdaptiveAvgPool2d((1, None))

        # Coordinate Attention Generation
        self.coordinate_conv = nn.Sequential( ConvBN(channels, internal_channels, kernel_size=1, 
                                                     stride=1, padding=0, bias=True),
                                              h_swish())   
            
        self.conv_h = nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    @staticmethod
    def get_module_name():
        return "CoordAtt"

    def forward(self, x):
        identity = x

        origin_shape = x.size() # B, C, H, W
        f_h, f_w = origin_shape[2], origin_shape[3]

        x_h = self.h_avg_pooling(x) # B, C, H, 1
        x_w = self.w_avg_pooling(x).permute(0, 1, 3, 2) # (B, C, 1, W) -> (B, C, W, 1)
        feat = torch.cat([x_h, x_w], dim=2) # B, C, W+H, 1

        attention_vectors = self.coordinate_conv(feat)
        attention_h, attention_w = torch.split(attention_vectors, [f_h, f_w], dim=2) # (B, internal_dims, W+H, 1) -> [(B, internal_dims, H, 1), (B, internal_dims, W, 1)] 
        
        a_h = self.conv_h(attention_h).sigmoid() # B, C, H, 1
        attention_w = attention_w.permute(0, 1, 3, 2)
        a_w = self.conv_w(attention_w).sigmoid() # B, C, 1, W

        return identity * a_w * a_h
    