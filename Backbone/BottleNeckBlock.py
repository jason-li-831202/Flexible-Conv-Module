import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Conv.DwsConv import DepthwiseSeparableConv2d
from Conv.GhostConv import GhostConv2d
from Transformer.utils import DropPath
from Attention.SEModule import SEBlock

class BottleNeck(nn.Module):
    def __init__(self,in_channels, out_channels, internal_expansion: float = 4, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1, dilation: int = 1, drop_path_rate: float = 0., use_nonlinear: bool = True):
        """ Inverted Residual Block

        1x1 Conv -> 3x3 DwConv -> 1x1 Conv

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_expansion: the ratio for internal_channels.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            drop_path_rate: Stochastic depth rate. Default: 0.0
            use_nonlinear: Whether to use activate function.
        """
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.padding =  (dilation*(kernel_size[0]-1)//2, dilation*(kernel_size[1]-1)//2)
        internal_channels = int(in_channels*internal_expansion)
        
        self.pw_conv = ConvBN(in_channels, internal_channels, kernel_size=1, stride=1, bias=False, 
                              activation=nn.ReLU(False))

        self.dws_conv = DepthwiseSeparableConv2d(internal_channels, out_channels, kernel_size=kernel_size, 
                                                 stride=self.stride, padding=self.padding, dilation=self.dilation, 
                                                 bias=False, use_nonlinear=True, activation=nn.ReLU(False))

        if(self.stride != 1 or in_channels != out_channels):
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.relu = nn.ReLU(False) if use_nonlinear else nn.Identity()

    def forward(self,x):
        residual = self.shortcut(x)
 
        out = self.pw_conv(x) # B, C, H, W
        out = self.dws_conv(out) # (B, C, H, W) -> (B, 4C, H, W)

        return self.relu(self.drop_path(out) + residual)
    
class MBottleNeck(nn.Module):
    def __init__(self,in_channels, out_channels, internal_expansion: float = 4, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1, dilation: int = 1, drop_path_rate: float = 0., use_nonlinear: bool = True):
        """ Mobile Inverted Residual Block

        1x1 Conv -> 3x3 DwConv -> SE Attention -> 1x1 Conv

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_expansion: the ratio for internal_channels.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            drop_path_rate: Stochastic depth rate. Default: 0.0
            use_nonlinear: Whether to use activate function.
        """
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.padding =  (dilation*(kernel_size[0]-1)//2, dilation*(kernel_size[1]-1)//2)
        internal_channels = int(in_channels*internal_expansion)
        
        self.pw_conv = ConvBN(in_channels, internal_channels, kernel_size=1, stride=1, bias=False, 
                              activation=nn.ReLU6(False))

        self.dws_se_conv = DepthwiseSeparableConv2d(internal_channels, out_channels, kernel_size=kernel_size, 
                                                 stride=self.stride, padding=self.padding, dilation=self.dilation, 
                                                 bias=False, use_se=True, use_nonlinear=True, activation=nn.ReLU6(inplace=True))

        if(self.stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.relu = nn.ReLU6(False) if use_nonlinear else nn.Identity()

    def forward(self,x):
        residual = self.shortcut(x)
 
        out = self.pw_conv(x) # B, C, H, W
        out = self.dws_se_conv(out) # (B, C, H, W) -> (B, 4C, H, W)

        return self.relu(self.drop_path(out) + residual)

class Fuse_MBottleNeck(nn.Module):

    def __init__(self,in_channels, out_channels, internal_expansion: float = 4, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1, dilation: int = 1, drop_path_rate: float = 0.):
        """ Fused Mobile Inverted Residual Block

        3x3 Conv -> SE Attention -> 1x1 Conv

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_expansion: the ratio for internal_channels.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            drop_path_rate: Stochastic depth rate. Default: 0.0
        """
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.padding =  (dilation*(kernel_size[0]-1)//2, dilation*(kernel_size[1]-1)//2)
        internal_channels = int(in_channels*internal_expansion)
  
        self.exp_conv = ConvBN(in_channels, internal_channels, kernel_size=kernel_size, stride=self.stride, dilation=dilation, 
                            groups=1, padding=self.padding, activation=nn.ReLU6(False))
        self.se = SEBlock(internal_channels, internal_channels)
        self.pw_conv = ConvBN(internal_channels, out_channels, kernel_size=1, stride=1, activation=nn.ReLU6(False))

        if(self.stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self,x):
        residual = self.shortcut(x)
 
        out = self.exp_conv(x)
        out = self.se(out)
        out = self.pw_conv(out)

        return self.drop_path(out) + residual
    
class GhostBottleNeck(nn.Module):

    def __init__(self,in_channels, out_channels, internal_expansion: float = 4, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1):
        """

        1x1 GhostConv -> 3x3 DwConv -> 1x1 GhostConv

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_expansion: the ratio for internal_channels.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
        """
        super().__init__()
        self.stride = stride
        internal_channels = int(in_channels*internal_expansion)
        
        self.pw_reduce_conv = GhostConv2d(in_channels, internal_channels, kernel_size=1, cheap_kernel_size=kernel_size,
                                          stride=1, bias=False, use_nonlinear=True)

        self.dw_conv = ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, 
                              stride=self.stride, bias=False, activation=nn.ReLU(True))

        self.pw_expansion_conv = GhostConv2d(internal_channels, out_channels, kernel_size=1, cheap_kernel_size=kernel_size,
                                             stride=1, bias=False, use_nonlinear=True)

        if(stride != 1 or in_channels != out_channels):
           self.shortcut = GhostConv2d(in_channels, out_channels, kernel_size=1, cheap_kernel_size=kernel_size,
                                         stride=self.stride, bias=False, use_nonlinear=False)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(False)

    def forward(self,x):
        residual = self.shortcut(x)

        out = self.pw_reduce_conv(x) # (B, C, H, W) -> (B, C_out//4, H, W)
        out = self.dw_conv(out) # (B, C_out//4, H, W)
        out = self.pw_expansion_conv(out) # (B, C_out//4, H, W) -> (B, C_out, H, W)

        return self.relu(out + residual)
    
class SandglassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, internal_expansion: float = 4, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1, dilation: int = 1, drop_path_rate: float = 0.):
        """

        3x3 DwConv -> 1x1 Conv -> 1x1 Conv -> 3x3 DwConv 

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_expansion: the ratio for internal_channels.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            drop_path_rate: Stochastic depth rate. Default: 0.0
        """
        super(SandglassBlock, self).__init__()
        kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.padding =  (dilation*(kernel_size[0]-1)//2, dilation*(kernel_size[1]-1)//2)
        internal_channels = int(in_channels*internal_expansion)

        self.bottle_neck = nn.Sequential(
            ConvBN(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=self.padding,
                   groups=in_channels, activation=nn.ReLU(inplace=True)),
            ConvBN(in_channels, internal_channels, kernel_size=1, stride=1, padding=0,
                   groups=1),
            ConvBN(internal_channels, out_channels, kernel_size=1, stride=1, padding=0,
                   groups=1, activation=nn.ReLU(inplace=True)),
            ConvBN(out_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation,
                   groups=out_channels)
        )

        if(self.stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.bottle_neck(x)
        return self.drop_path(out) + residual
  
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input=torch.randn(24, 32, 112, 112).to(device)
    a = MBottleNeck(32, 64, kernel_size=3, stride=2)

    a.to(device)
    a.eval()
    out = a(input)
    save_model_to_onnx(a.to(device), input, "ori_model.onnx") 