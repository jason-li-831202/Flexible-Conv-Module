import torch
from torch import mean, nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Union, Tuple, Optional

# CondConv: Conditionally Parameterized Convolutions for Efficient Inference (https://arxiv.org/abs/1904.04971)
# NeurIPS 2019

class Attention(nn.Module):
    def __init__(self, in_channels, num_experts, init_weight = True):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, num_experts, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if(init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        attn = self.global_avg_pool(x) # B, C, 1, 1
        attn = self.fc(attn).view(x.shape[0], -1) # B, num_experts
        return self.sigmoid(attn)

class CondConv2d(_ConvNd):
    r"""Learn specialized convolutional kernels for each example.

    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv), 
    which challenge the paradigm of static convolutional kernels 
    by computing convolutional kernels as a function of the input.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer 
    """

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'num_experts', 'init_weight']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self,  in_channels: int,  
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]], 
                        stride: int = 1, 
                        padding: Union[int, Tuple[int, int]] = 0, 
                        dilation: int = 1, 
                        groups: int = 1, 
                        bias: bool = True, 
                        padding_mode: str = 'zeros', 
                        num_experts: int = 3, 
                        init_weight: bool = True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.num_experts = num_experts
        self.attn = Attention(in_channels, num_experts, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(
            num_experts, out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if(bias):
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels), requires_grad=True)
        else:
            self.bias = None

        if(init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.weight[i])

    def _conv_forward(self, bs, inputs, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(inputs, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight = weight,
                            bias = bias,
                            stride = self.stride,
                            padding = _pair(0), 
                            dilation = self.dilation,
                            groups = self.groups*bs)
        else :
            return F.conv2d(inputs, 
                            weight = weight,
                            bias = bias,
                            stride = self.stride,
                            padding = self.padding,
                            dilation = self.dilation,
                            groups = self.groups*bs)
        
    def forward(self, inputs):
        B, C, H, W = inputs.shape
        routing_weights = self.attn(inputs) # bs, num_experts

        inputs = inputs.view(1, -1, H, W)
        weight = self.weight.view(self.num_experts, -1) # num_experts, -1
        aggregate_weight = torch.mm(routing_weights, weight).view(B*self.out_channels,
                                                                  self.in_channels // self.groups,
                                                                  *self.kernel_size) # (bs*out_c, in_c, k, k)
        if(self.bias is not None):
            bias = self.bias.view(self.num_experts, -1) # num_experts, out_c
            aggregate_bias = torch.mm(routing_weights, bias).view(-1) # bs, out_c
        else :
            aggregate_bias = None
        
        out = self._conv_forward(B, inputs, aggregate_weight, aggregate_bias)
        return out.view(B, self.out_channels, out.size(-2), out.size(-1))