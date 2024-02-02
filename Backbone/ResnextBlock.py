import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Backbone.BottleNeckBlock import BottleNeck

__all__ = ["ResnextUnit", "ResnextStage"]
# Aggregated Residual Transformations for Deep Neural Networks (https://arxiv.org/abs/1611.05431)
# CVPR 2017

class ResnextUnit(BottleNeck):
    def __init__(self, in_channels, out_channels, internal_expansion: float = 2, kernel_size: Union[int, Tuple[int, int]] = 3, groups: int = 32,
                 stride: int = 1, dilation: int = 1, drop_path_rate: float = 0., use_nonlinear: bool = True):
        super(ResnextUnit, self).__init__(in_channels, out_channels, internal_expansion, kernel_size, stride, dilation, drop_path_rate, use_nonlinear)
        self.__delattr__("dws_conv")
        internal_channels =int(in_channels*internal_expansion)
        
        self.pw_conv.conv.stride = self.stride
        self.group_conv = nn.Sequential(
            ConvBN(internal_channels, internal_channels, kernel_size=kernel_size, 
                    stride=1, padding=self.padding, dilation=self.dilation, groups=groups,
                    bias=False, activation=nn.ReLU(False) if use_nonlinear else None),

            ConvBN(internal_channels, out_channels, kernel_size=1, 
                   bias=False, activation=nn.ReLU(False) if use_nonlinear else None),        
        )

    def forward(self,x):
        residual = self.shortcut(x)

        out = self.pw_conv(x) # B, C, H, W
        out = self.group_conv(out)

        return self.relu(self.drop_path(out) + residual)
    
class ResnextStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_units: int, kernel_size: Union[float, Tuple[float, float]]  = (3, 3), 
                 stride: int = 2, dilation: int = 1, groups: int = 32, use_nonlinear: bool = True) :
        """ Build a stage of Resnext model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            use_nonlinear: Whether to use activate function.
        """
        super(ResnextStage, self).__init__()

        units = [ResnextUnit(in_channels, out_channels, internal_expansion=2/stride, kernel_size=kernel_size, 
                                     stride=stride, dilation=dilation, groups=groups, use_nonlinear=use_nonlinear)]
        for i in range(num_units-1):
            units.append(ResnextUnit(out_channels, out_channels, internal_expansion=0.5, kernel_size=kernel_size, 
                                     stride=1, dilation=dilation, groups=groups, use_nonlinear=use_nonlinear))

        self.resnext_stage = nn.Sequential(*units)

    def forward(self, x):
        return self.resnext_stage(x)
    
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [256, 3, 1, 32],
        [512, 4, 2, 32],
        [1024, 6, 2, 32],
        [2048, 3, 2, 32],
    ]
    input=torch.randn(3, 64, 56, 56).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        resnext = ResnextStage(input.size(1), opt[0], num_units=opt[1], stride=opt[2], groups=opt[3])
        resnext.to(device)
        resnext.eval()
        
        # ori result
        out = resnext(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(resnext, input)
        summary(copy.deepcopy(resnext).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(resnext.to("cpu"), input.to("cpu"), f"resnext_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")