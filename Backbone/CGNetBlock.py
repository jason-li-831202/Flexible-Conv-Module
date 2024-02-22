import torch, sys
from torch import nn
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Attention.SEModule import SEBlock

__all__ = ["ContextGuidedUnit", "ContextGuidedStage"]
# CGNet: A Light-weight Context Guided Network for Semantic Segmentation (https://arxiv.org/pdf/1811.08201)
# CVPR 2019

class ContextGuidedUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation_rate: int = 2, reduction: int = 16):
        super().__init__()
        internal_channels =  out_channels if stride >= 2 else int(out_channels//2)
        
        self.conv = ConvBN(in_channels, internal_channels, kernel_size=kernel_size if stride >= 2 else 1, stride=stride, 
                           padding=kernel_size//2 if stride >= 2 else 0, activation=nn.PReLU()) 
        self.loc_conv = nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, 
                                  groups=internal_channels, bias=False)
        self.sur_conv = nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=dilation_rate*(kernel_size-1)//2, 
                                  dilation=dilation_rate, groups=internal_channels, bias=False)

        self.activation = nn.Sequential(
            nn.BatchNorm2d(2*internal_channels),
            nn.PReLU())
        self.shortcut = nn.Identity() if stride == 1 else None
        self.reduce = nn.Conv2d(2*internal_channels, out_channels, kernel_size=1, stride=1) if stride >= 2 else nn.Identity()

        self.se = SEBlock(out_channels, reduction)

    def forward(self, input):
        output = self.conv(input)
        loc = self.loc_conv(output)
        sur = self.sur_conv(output)
        feats = torch.cat([loc, sur], 1) 

        feats = self.activation(feats)
        feats = self.reduce(feats)

        output = self.se(feats)
        if self.shortcut != None:
            output  = input + output
        return output
    
class ContextGuidedStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_units: int, kernel_size: int = 3, 
                 dilation_rate: int = 2, reduction: int = 16):
        """ Build a stage of ContextGuidedStage model.
        Args:
            in_channels: input channel dimensionality.
            out_channels: Number of channels produced by the convolution.
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            dilation_rate: Spacing between kernel elements.
            reduction: the ratio for SEBlock internal_channels.
        """
        super().__init__()
        self.downsample = ContextGuidedUnit(in_channels, out_channels, kernel_size=kernel_size, stride=2, 
                                            dilation_rate=dilation_rate, reduction=reduction)

        units = []
        for i in range(num_units - 1):
            units.append(ContextGuidedUnit(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                                           dilation_rate=dilation_rate, reduction=reduction))
        self.cg_stage = nn.ModuleList(units)

    def forward(self, x):
        part1 = self.downsample(x)
        for i, unit in enumerate(self.cg_stage):
            part2 = unit(part1 if i==0 else part2)

        return torch.cat([part1, part2], dim=1)
    
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [64, 3, 2, 8],
        [128, 21, 4, 16],
    ]
    input=torch.randn(1, 35, 56, 56).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        cgnet = ContextGuidedStage(input.size(1), out_channels=opt[0], num_units=opt[1], dilation_rate=opt[2],
                                    reduction=opt[3])
        cgnet.to(device)
        cgnet.eval()
        
        # ori result
        out = cgnet(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(cgnet, input)
        summary(copy.deepcopy(cgnet).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(cgnet.to("cpu"), input.to("cpu"), f"pelee_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")
