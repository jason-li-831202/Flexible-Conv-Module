import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Backbone.BottleNeckBlock import SandglassBlock

__all__ = ["MobilenextStage"]
# Rethinking Bottleneck Structure for Efficient Mobile Network Design(https://arxiv.org/abs/2007.02269)
# ECCV 2020

class MobilenextStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_units: int, kernel_size: Union[float, Tuple[float, float]]  = (3, 3), 
                 internal_reduction: int = 6, stride: int = 2, dilation: int = 1) :
        """ Build a stage of Mobilenext model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            internal_reduction: .Default: 6
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
        """
        super(MobilenextStage, self).__init__()

        units = [SandglassBlock(in_channels, out_channels, internal_expansion=1/internal_reduction, kernel_size=kernel_size, 
                                     stride=stride, dilation=dilation)]
        for i in range(num_units-1):
            units.append(SandglassBlock(out_channels, out_channels, internal_expansion=1/internal_reduction, kernel_size=kernel_size, 
                                     stride=1, dilation=dilation))

        self.mobilenext_stage = nn.Sequential(*units)

    def forward(self, x):
        return self.mobilenext_stage(x)
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        # out_channels, blocks, reduction, stride
        [96,   1, 2, 2],
        [144,  1, 6, 1],
        [192,  3, 6, 2],
        [288,  3, 6, 2],
        [384,  4, 6, 1],
        [576,  4, 6, 2],
        [960,  2, 6, 1],
        [1280, 1, 6, 1]
    ]
    input=torch.randn(3, 32, 56, 56).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        mobilenext = MobilenextStage(input.size(1), opt[0], num_units=opt[1], 
                                     internal_reduction=opt[2], 
                                     stride=opt[3])
        mobilenext.to(device)
        mobilenext.eval()
        
        # ori result
        out = mobilenext(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(mobilenext, input)
        summary(copy.deepcopy(mobilenext).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(mobilenext.to("cpu"), input.to("cpu"), f"mobilenext_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")
