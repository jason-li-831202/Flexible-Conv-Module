import torch, sys
from torch import nn
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Attention.PSAModule import PSABlock
from Backbone.BottleNeckBlock import BottleNeck

__all__ = ["EPSAUnit", "EPSAStage"]
# EPSANet: An Efficient Pyramid Squeeze Attention Block on Convolutional Neural Network (https://arxiv.org/abs/2105.14447)
# CVPR 2021

class EPSAUnit(BottleNeck):
    def __init__(self, in_channels, out_channels, internal_expansion: float = 0.25, multi_kernels: list = [3, 5, 7, 9],
                 multi_groups: list = [1, 4, 8, 16], stride: int = 1, drop_path_rate: float = 0., use_nonlinear: bool = True):
        super(EPSAUnit, self).__init__(in_channels, out_channels, internal_expansion, 3, stride, 1, drop_path_rate, use_nonlinear)
        self.__delattr__("dws_conv")
        internal_channels =int(in_channels*internal_expansion)
        
        self.psa_conv = nn.Sequential(
            PSABlock(internal_channels, internal_channels, multi_kernels=multi_kernels, 
                    multi_groups=multi_groups, stride=self.stride),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(False),

            ConvBN(internal_channels, out_channels, kernel_size=1, 
                   bias=False, activation=nn.ReLU(False) if use_nonlinear else None),        
        )

    def forward(self,x):
        residual = self.shortcut(x)

        out = self.pw_conv(x) # B, C, H, W
        out = self.psa_conv(out)

        return self.relu(self.drop_path(out) + residual)
    
class EPSAStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_units: int, multi_kernels: list = [3, 5, 7, 9], multi_groups: list = [1, 4, 8, 16], 
                 stride: int = 2, use_nonlinear: bool = True) :
        """ Build a stage of EPSA model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            multi_kernels: Spatial size of the kernel for each pyramid level
            multi_groups: Number of blocked connections from input channels to output channels for each pyramid level
            stride: Stride of the convolution. Default: 1
            use_nonlinear: Whether to use activate function.
        """
        super(EPSAStage, self).__init__()

        units = [EPSAUnit(in_channels, out_channels, internal_expansion=1/stride, multi_kernels=multi_kernels, multi_groups=multi_groups,
                                     stride=stride, use_nonlinear=use_nonlinear)]
        for i in range(num_units-1):
            units.append(EPSAUnit(out_channels, out_channels, internal_expansion=0.25, multi_kernels=multi_kernels, multi_groups=multi_groups,
                                     stride=1, use_nonlinear=use_nonlinear))

        self.epsanet_stage = nn.Sequential(*units)

    def forward(self, x):
        return self.epsanet_stage(x)
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [256, 3, 1],
        [512, 4, 2],
        [1024, 6, 2],
        [2048, 3, 2],
    ]
    input=torch.randn(3, 64, 14, 14).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        epsa = EPSAStage(input.size(1), opt[0], num_units=opt[1], stride=opt[2])
        epsa.to(device)
        epsa.eval()
        
        # ori result
        out = epsa(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(epsa, input)
        summary(copy.deepcopy(epsa).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(epsa.to("cpu"), input.to("cpu"), f"epsa_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")