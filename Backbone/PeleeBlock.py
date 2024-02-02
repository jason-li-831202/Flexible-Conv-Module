import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Transformer.utils import DropPath

__all__ = ["PeleeUnit", "PeleeStage"]
# Pelee: A Real-Time Object Detection System on Mobile Devices (https://arxiv.org/abs/1804.06882)
# CVPR 2019

class PeleeUnit(nn.Module):
    def __init__(self, in_channels, out_channels, internal_expansion: float = 2, kernel_size: Union[int, Tuple[int, int]] = 3,
                 drop_path_rate: float = 0., use_nonlinear: bool = True):
        super().__init__()
        internal_channel = out_channels*internal_expansion
        kernel_size = _pair(kernel_size)
        padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2)

        self.part1_conv = nn.Sequential(OrderedDict([
            ("pw", ConvBN(in_channels, internal_channel, kernel_size=1, stride=1, bias=False, 
                          activation=nn.ReLU(True) if use_nonlinear else nn.Identity() )),
            ("convkxk", ConvBN(internal_channel, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, 
                               activation=nn.ReLU(True) if use_nonlinear else nn.Identity() ))
        ]))

        self.part2_conv = nn.Sequential(OrderedDict([
            ("pw", ConvBN(in_channels, internal_channel, kernel_size=1, stride=1, bias=False, 
                          activation=nn.ReLU(True) if use_nonlinear else nn.Identity() )),
            ("conv1kxk", ConvBN(internal_channel, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False,
                               activation=nn.ReLU(True) if use_nonlinear else nn.Identity() )),
            ("conv2kxk", ConvBN(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False,
                               activation=nn.ReLU(True) if use_nonlinear else nn.Identity() )),
        ]))
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        part1_out = self.part1_conv(x)
        part2_out = self.part2_conv(x)

        return torch.cat((x, part1_out, part2_out),dim=1)
    
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_pooling = True):
        super(TransitionBlock, self).__init__()
        self.trans_conv = nn.Sequential(
            ConvBN(in_channels, out_channels, kernel_size=1, stride=1, bias=False,
                   activation=nn.ReLU(False)),
            nn.AvgPool2d(kernel_size=2, stride=2) if with_pooling else nn.Identity()
        )

    def forward(self, x):
        out = self.trans_conv(x)
        return out
    
class PeleeStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_units: int, bottleneck_width: int, 
                 kernel_size: Union[float, Tuple[float, float]]  = (3, 3), with_pooling: bool = False, use_nonlinear: bool = True) :
        """ Build a stage of Pelee model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            use_nonlinear: Whether to use activate function.
        """
        super(PeleeStage, self).__init__()
        units = []
        for i in range(num_units):
            units.append(PeleeUnit(in_channels, 16, internal_expansion=bottleneck_width, kernel_size=kernel_size, 
                                   use_nonlinear=use_nonlinear))
            in_channels += 16 * 2

        #Transition Layer without Compression
        units.append(TransitionBlock(in_channels, out_channels, with_pooling))

        self.pelee_stage = nn.Sequential(*units)

    def forward(self, x):
        return self.pelee_stage(x)
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [128, 3, 1, True],
        [256, 4, 2, True],
        [512, 8, 4, True],
        [704, 6, 4, False],
    ]
    input=torch.randn(1, 32, 56, 56).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        peleenet = PeleeStage(input.size(1), out_channels=opt[0], num_units=opt[1], bottleneck_width=opt[2],
                              with_pooling=opt[3])
        peleenet.to(device)
        peleenet.eval()
        
        # ori result
        out = peleenet(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(peleenet, input)
        summary(copy.deepcopy(peleenet).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(peleenet.to("cpu"), input.to("cpu"), f"pelee_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")
