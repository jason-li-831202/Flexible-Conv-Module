import torch, math
from torch import nn
from torch.nn import functional as F

from Conv.Basic.common import ConvBN

__all__ = ["RFBBlock", "RFBsBlock"]
# Receptive Field Block Net for Accurate and Fast Object Detection (https://arxiv.org/abs/1711.07767)
# ECCV 2018
class RFBBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, 
                 min_dilation: int = 1, use_bias: bool = False, internal_reduce: int = 8, scale = 0.1):
        """
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution.
            min_dilation: Minimum visual field.
            use_bias: Whether to use conv bias.
            internal_reduce: the ratio for internal_channels.
        """
        super(RFBBlock, self).__init__()
        assert min_dilation%2 !=0, "min_dilation must be an odd number."
        internal_padding = (kernel_size-1)//2
        internal_channels = in_channels // internal_reduce
        self.scale = scale
        
        # branch 0
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=use_bias)

        # branch 1（1x1 conv -> kxk conv,r=d）
        branch_padding = min_dilation*internal_padding
        self.branch1 = nn.Sequential(
            ConvBN(in_channels, 2*internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(2*internal_channels, 2*internal_channels, kernel_size=kernel_size, 
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )
        
        # branch 2 （1x1 conv -> kxk conv -> kxk conv,r=d+2）
        min_dilation += 2
        branch_padding = min_dilation*internal_padding
        self.branch2 = nn.Sequential(
            ConvBN(in_channels, internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, 2*internal_channels, kernel_size=kernel_size, 
                   stride=1, padding=internal_padding, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(2*internal_channels, 2*internal_channels, kernel_size=kernel_size, 
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )
        
        # branch 3 （1x1 conv -> (k+2)x(k+2) conv -> kxk conv,r=d+4）
        min_dilation += 2
        branch_padding = min_dilation*internal_padding
        self.branch3 = nn.Sequential(
            ConvBN(in_channels, internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            # Note: 5x5 conv拆分成两个3x3 conv ---------------------------------------------
            ConvBN(internal_channels, (internal_channels//2)*3, kernel_size=kernel_size, 
                stride=1, padding=internal_padding, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN((internal_channels//2)*3, 2*internal_channels, kernel_size=kernel_size, 
                stride=1, padding=internal_padding, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            # -----------------------------------------------------------------------------    
            ConvBN(2*internal_channels, 2*internal_channels, kernel_size=kernel_size, 
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )

        self.conv_linear = nn.Conv2d(6 * internal_channels, out_channels, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.identity(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch1, branch2, branch3], 1)
        out = self.conv_linear(out)

        out = out + shortcut*self.scale
        return self.activation(out)
    

class RFBsBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, 
                 min_dilation: int = 1, use_bias: bool = False, internal_reduce: int = 4, scale: float = 0.1):
        """
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution.
            min_dilation: Minimum visual field.
            use_bias: Whether to use conv bias.
            internal_reduce: the ratio for internal_channels.
        """

        super(RFBsBlock, self).__init__()
        assert min_dilation%2 !=0, "min_dilation must be an odd number."
        internal_padding = (kernel_size-1)//2
        internal_channels = in_channels // internal_reduce
        self.scale = scale

        # branch 0
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=use_bias)

        # branch 1（1x1 conv -> kxk conv,r=d）
        branch_padding = min_dilation*internal_padding
        self.branch1 = nn.Sequential(
            ConvBN(in_channels, internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, 
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias),
        )

        min_dilation += 2
        branch_padding = min_dilation*internal_padding
        # branch 2（1x1 conv -> 1xk conv -> kxk conv,r=d+2）
        self.branch2 = nn.Sequential(
            ConvBN(in_channels, internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=(1, kernel_size), 
                   stride=1, padding=(0, internal_padding), bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, 
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )

        # branch 3 （1x1 conv -> kx1 conv -> kxk conv,r=d+2）
        self.branch3 = nn.Sequential(
            ConvBN(in_channels, internal_channels, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=(kernel_size, 1), 
                   stride=1, padding=(internal_padding, 0), bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=kernel_size,  
                    stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )

        min_dilation += 2
        branch_padding = min_dilation*internal_padding
        # branch 4 （1x1 conv -> 1xk conv -> kx1 conv -> kxk conv,r=d+4）
        self.branch4 = nn.Sequential(
            ConvBN(in_channels, internal_channels // 2, kernel_size=1, 
                   stride=1, padding=0, bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels // 2, (internal_channels // 4) * 3, kernel_size=(1, kernel_size), 
                   stride=1, padding=(0, internal_padding), bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN((internal_channels // 4) * 3, internal_channels, kernel_size=(kernel_size, 1), 
                   stride=1, padding=(internal_padding, 0), bias=use_bias, activation=nn.LeakyReLU(0.2, True)),
            ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, 
                   stride=stride, padding=branch_padding, dilation=min_dilation, bias=use_bias)
        )

        self.conv_linear = nn.Conv2d(4 * internal_channels, out_channels, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.identity(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.conv_linear(out)

        out = out + shortcut*self.scale
        return self.activation(out)