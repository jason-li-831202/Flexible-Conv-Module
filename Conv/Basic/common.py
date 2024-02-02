from typing import Union, Tuple
from torch import mean, nn

def _ntuple(n):
    def parse(x):
        if not isinstance(x, tuple):
            from itertools import repeat
            return tuple(repeat(x, n))
        return x

    return parse

_pair = _ntuple(2)

class ConvBN(nn.Module):
    def __init__(self, input_channel: int, 
                       output_channel: int, 
                       kernel_size: Union[int, Tuple[int, int]] = 3, 
                       stride: int = 1, 
                       padding: Union[int, Tuple[int, int]] = 0, 
                       dilation: int = 1, 
                       groups: int = 1, 
                       bias: bool = False,
                       activation: nn = None):
        """Initialize Conv layer with given arguments including activation."""
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, padding_mode='zeros', groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(output_channel)
        self.activation = activation if activation!=None else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.activation(self.bn(self.conv(x)))