import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Attention.SEModule import SEBlock

__all__ = ["VarGroupConv", "VarGroupBlock", "VarGroupBlock_DownSampling", "VarGroupStage"]
# VarGFaceNet: An Efficient Variable Group Convolutional Neural Network for Lightweight Face Recognition (https://arxiv.org/pdf/1910.04985)
# ICCV 2019

class VarGroupConv(ConvBN):
    def __init__(self, in_channels, out_channels, kernel_size, groups_reduce, stride=1, dilation=1, isPRelu=False):
        super().__init__(input_channel=in_channels, 
                        output_channel=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding= (dilation*(_pair(kernel_size)[0]-1)//2, dilation*(_pair(kernel_size)[1]-1)//2),
                        groups=in_channels//groups_reduce,
                        bias=False,
                        activation=nn.PReLU() if isPRelu else None)

class VarGroupBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: Union[int, Tuple[int, int]], dilation: int = 1, groups_reduce: int = 8):
        """ 
        Args:
            in_channels: Number of channels in the input image
            kernel_size: Size of the conv kernel
            dilation: Spacing between kernel elements. Default: 1
            groups_reduce: reduce rate of blocked connections from input channels to output channels. Default: 8
        """
        super(VarGroupBlock, self).__init__()
        stride = 1
        internal_channels = 2 * in_channels

        self.varG1_conv = nn.Sequential(
            VarGroupConv(in_channels, internal_channels, kernel_size, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation),
            VarGroupConv(internal_channels, in_channels, 1, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation, isPRelu=True)
        )

        self.varG2_conv = nn.Sequential(
            VarGroupConv(in_channels, internal_channels, kernel_size, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation),
            VarGroupConv(internal_channels, in_channels, 1, 
                         groups_reduce=groups_reduce, stride=stride, dilation=dilation, isPRelu=False)
        )

        if(stride != 1 ):
            self.shortcut = VarGroupConv(in_channels, in_channels, kernel_size=1, 
                                         groups_reduce=groups_reduce, stride=stride)
        else:
            self.shortcut = nn.Identity()

        self.se =  SEBlock(in_channels, in_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.varG1_conv(x)
        x = self.varG2_conv(x)
        out = self.se(x)

        return self.prelu(out + residual)

class VarGroupBlock_DownSampling(nn.Module):
    def __init__(self, in_channels: int, kernel_size: Union[int, Tuple[int, int]], dilation: int = 1, groups_reduce: int = 8):
        """ 
        Args:
            in_channels: Number of channels in the input image
            kernel_size: Size of the conv kernel
            dilation: Spacing between kernel elements. Default: 1
            groups_reduce: reduce rate of blocked connections from input channels to output channels. Default: 8
        """
        super(VarGroupBlock_DownSampling, self).__init__()
        stride = 2
        internal_channels = 2 * in_channels

        self.varG_branch1_conv = nn.Sequential(
            VarGroupConv(in_channels, internal_channels, kernel_size, 
                         groups_reduce=groups_reduce, stride=stride, dilation=dilation),
            VarGroupConv(internal_channels, internal_channels, 1, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation, isPRelu=True)
        )

        self.varG_branch2_conv = nn.Sequential(
            VarGroupConv(in_channels, internal_channels, kernel_size, 
                         groups_reduce=groups_reduce, stride=stride, dilation=dilation),
            VarGroupConv(internal_channels, internal_channels, 1, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation, isPRelu=True)
        )


        self.varG_concat_conv = nn.Sequential(
            VarGroupConv(internal_channels, internal_channels*2, kernel_size, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation),
            VarGroupConv(internal_channels*2, internal_channels, 1, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation, isPRelu=False)
        )

        self.shortcut = nn.Sequential(
            VarGroupConv(in_channels, internal_channels, kernel_size, 
                         groups_reduce=groups_reduce, stride=stride, dilation=dilation),
            VarGroupConv(internal_channels, internal_channels, 1, 
                         groups_reduce=groups_reduce, stride=1, dilation=dilation, isPRelu=False)
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.shortcut(x)

        x1 = self.varG_branch1_conv(x)
        x2 = self.varG_branch2_conv(x)
        concat_x = x1 + x2
        out = self.varG_concat_conv(concat_x)

        return self.prelu(out + residual)

class VarGroupStage(nn.Module):
    def __init__(self, in_channels: int, num_units: int, kernel_size: Union[int, Tuple[int, int]], dilation: int = 1, groups_reduce: int = 8):
        """ Build a stage of VarGroup model.
        Args:
            in_channels: Number of channels in the input image
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            dilation: Spacing between kernel elements. Default: 1
        """
        super(VarGroupStage, self).__init__()

        units = [VarGroupBlock_DownSampling(in_channels, kernel_size=kernel_size, dilation=dilation, groups_reduce=groups_reduce)]
        in_channels *= 2
        for i in range(num_units-1):
            units.append(VarGroupBlock(in_channels, kernel_size=kernel_size, dilation=dilation, groups_reduce=groups_reduce))
        self.vargroup_stage = nn.Sequential(*units)

    def forward(self, x):
        return self.vargroup_stage(x)
    
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [3, 3, 1, 8],
        [7, 3, 1, 8],
        [4, (3, 3), 1, 8],
    ]
    input=torch.randn(3, 40, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        vargroup = VarGroupStage(input.size(1), num_units=opt[0], kernel_size=opt[1], dilation=opt[2], groups_reduce=opt[3])
        vargroup.to(device)
        vargroup.eval()
        
        # ori result
        out = vargroup(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(vargroup, input)
        summary(copy.deepcopy(vargroup).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(vargroup, input, f"vargroup_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")

