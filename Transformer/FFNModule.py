import torch, math
from torch import nn

from Transformer.utils import DropPath, LayerNorm, view_format
from Conv.Basic.common import ConvBN

__all__ = ["FFNBlock_Conv", "FFNBlock_FC", "MixFFNBlock_FC", "InvertedResidualFFNBlock_Conv"]

# The common FFN Block used in many Transformer and MLP models.
class FFNBlock_Conv(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int=None, out_channels: int=None, 
                 use_nonlinear: bool=True, drop_path_rate: float=0.):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = internal_channels or in_channels

        self.ffn_pre_bn = nn.BatchNorm2d(in_channels)
        self.ffn_pw1 = ConvBN(in_channels, hidden_features, kernel_size=1, 
                              stride=1, padding=0, groups=1)
        self.ffn_pw2 = ConvBN(hidden_features, out_features, kernel_size=1, 
                              stride=1, padding=0, groups=1)
        self.activation = nn.GELU() if use_nonlinear else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, in_format="BCHW", out_format = "BCHW"): # (B, C, H, W)
        x = view_format(x, in_format, "BCHW")
        out = self.ffn_pre_bn(x)
        out = self.ffn_pw1(out)
        out = self.activation(out)
        out = self.ffn_pw2(out)
        return view_format(self.drop_path(out), "BCHW", out_format)


class FFNBlock_FC(nn.Module):
    def __init__(self, in_dims: int, internal_dims: int=None, out_dims: int=None, 
                 use_nonlinear: bool=True, drop_path_rate: float=0.):
        super().__init__()
        out_dims = out_dims or in_dims
        hidden_dims = internal_dims or in_dims

        self.ffn_pre_bn = LayerNorm(in_dims, eps=1e-6)
        self.ffn_fc1 = nn.Linear(in_dims, hidden_dims) # pointwise/1x1 convs, implemented with linear layers
        self.ffn_fc2 = nn.Linear(hidden_dims, in_dims)
        self.activation = nn.GELU() if use_nonlinear else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, in_format="BCHW", out_format = "BCHW"): # (B, C, H, W)
        if (in_format != "BNC") :
            x, shape = view_format(x, in_format, "BNC")
        else :
            shape = None
        out = self.ffn_pre_bn(x)
        out = self.ffn_fc1(out)
        out = self.activation(out)
        out = self.ffn_fc2(out)
        return view_format(self.drop_path(out), "BNC", out_format, shape=shape)


class MixFFNBlock_FC(nn.Module):
    def __init__(self, in_dims: int, internal_dims: int=None, out_dims: int=None, 
                 use_nonlinear: bool=True, drop_path_rate: float=0.):
        super().__init__()
        out_dims = out_dims or in_dims
        hidden_dims = internal_dims or in_dims

        self.ffn_pre_bn = LayerNorm(in_dims, eps=1e-6)
        self.ffn_fc1 = nn.Linear(in_dims, hidden_dims) # pointwise/1x1 convs, implemented with linear layers
        self.ffn_dw_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, stride=1, padding=1, 
                                     groups=hidden_dims, bias=True)
        self.ffn_fc2 = nn.Linear(hidden_dims, out_dims)
        self.activation = nn.GELU() if use_nonlinear else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x): # (B, H, W, C)
        out = self.ffn_pre_bn(x)
        out = self.ffn_fc1(out) 

        transpose = out.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        transpose = self.ffn_dw_conv(transpose) 
        out = transpose.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

        out = self.activation(out)
        out = self.ffn_fc2(out)
        return self.drop_path(out)
    

class InvertedResidualFFNBlock_Conv(nn.Module):
    def __init__(self, in_dims: int, internal_dims: int=None, out_dims: int=None, 
                 use_nonlinear: bool=True, drop_rate: float=0.):
        '''
            default: internal_dims = in_dims*4
        '''
        super(InvertedResidualFFNBlock_Conv, self).__init__()
        out_dims = out_dims or in_dims
        hidden_dims = internal_dims or in_dims

        self.ffn_pre_bn = LayerNorm(in_dims, eps=1e-6)
        self.ffn_pw1 = nn.Sequential( nn.Conv2d(in_channel=in_dims, out_channel=hidden_dims, kernel_size=1, stride=1, padding=0),
                                      nn.GELU(),
                                      nn.BatchNorm2d(hidden_dims))

        self.ffn_dw_conv = nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, stride=1, padding=1, 
                                     groups=hidden_dims, bias=True)
        if use_nonlinear :
            self.activation = nn.Sequential( nn.GELU(),
                                             nn.BatchNorm2d(hidden_dims))
        else :
            self.activation =nn.Identity()
        self.ffn_pw2 = ConvBN(hidden_dims, out_dims, kernel_size=1, stride=1, padding=0, groups=1)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x): # (B, H, W, C)
        x = self.ffn_pre_bn(x)

        xt = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        xt = self.ffn_pw1(xt)
        xt = self.drop(xt)

        xt = xt + self.activation(self.ffn_dw_conv(xt))

        xt = self.ffn_pw2(xt)
        xt = self.drop(xt)
        out = xt.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        return out 