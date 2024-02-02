import torch, sys
from torch import nn
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Conv.DwsConv import DepthwiseSeparableConv2d
from Attention.SEModule import ESEBlock_Conv

__all__ = ["OSAUnit", "DWsOSAUnit", "OSAStage"]
# An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection ( https://arxiv.org/abs/1904.09730 )
# CVPR 2019 

class OSAUnit(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3, 
                 num_residuals: int = 5, dilation: int = 1, use_ese: bool = False, use_identity: bool = False):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_reduce: Number of channels for residual.
            kernel_size: Size of the conv kernel
            num_residuals: Number of residual in this unit.
            dilation: Spacing between kernel elements. Default: 1
            use_ese: Whether to use ese attention.
            use_identity: Whether to use identity.
        """
        super(OSAUnit, self).__init__()
        self.dilation = dilation
        self.padding =  dilation*(kernel_size-1)//2

        self.identity = None
        if use_identity:
            self.identity = ConvBN(in_channels, out_channels, kernel_size=1, bias=False, activation=nn.ReLU(inplace=True)) \
                            if in_channels != out_channels else nn.Identity()

        self.residual_layers = nn.ModuleList()
        stage_channels = in_channels
        for i in range(num_residuals):
            self.residual_layers.append(ConvBN(stage_channels, internal_channels, kernel_size=kernel_size, 
                                       stride=1, padding=self.padding, dilation=self.dilation, bias=False, 
                                       activation=nn.ReLU(inplace=True)))
            stage_channels = internal_channels

        self.concat_conv = ConvBN(in_channels+internal_channels*num_residuals, out_channels, kernel_size=1,
                                  stride=1, bias=False, activation=nn.ReLU(inplace=True))
        
        self.ese = ESEBlock_Conv(out_channels) if use_ese else nn.Identity()

    def forward(self, x):
        residual = x
        feats = [x]
        for layer in self.residual_layers:
            x = layer(x)
            feats.append(x)
        concat_feat = torch.cat(feats, dim = 1)

        out = self.concat_conv(concat_feat)
        out = self.ese(out)

        if(self.identity != None):
            out += self.identity(residual)
        return out
    
class DWsOSAUnit(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, kernel_size: int = 3, 
                 num_residuals: int = 5, dilation: int = 1, use_ese: bool = False, use_identity: bool = False):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            internal_reduce: Number of channels for residual.
            kernel_size: Size of the conv kernel
            num_residuals: Number of residual in this unit.
            dilation: Spacing between kernel elements. Default: 1
            use_ese: Whether to use ese attention.
            use_identity: Whether to use identity.
        """
        super(DWsOSAUnit, self).__init__()
        self.dilation = dilation
        self.padding =  dilation*(kernel_size-1)//2

        self.identity = None
        if use_identity:
            self.identity = ConvBN(in_channels, out_channels, kernel_size=1, 
                                   bias=False, activation=nn.ReLU(inplace=True)) \
                                   if in_channels != out_channels else nn.Identity()


        self.reduce_conv = nn.Sequential(ConvBN(in_channels, internal_channels, kernel_size=1, 
                                                bias=False, activation=nn.ReLU(inplace=True)) ) \
                                                if in_channels != internal_channels else nn.Identity() 
        self.residual_layers = nn.ModuleList()
        for i in range(num_residuals):
            self.residual_layers.append(DepthwiseSeparableConv2d(internal_channels, internal_channels, kernel_size=kernel_size, 
                                                                stride=1, padding=self.padding, dilation=self.dilation, bias=False, 
                                                                use_bn=(False, True), use_nonlinear=(False, True) ))

        # feature aggregation
        self.concat_conv = ConvBN(in_channels+internal_channels*num_residuals, out_channels, kernel_size=1,
                                  stride=1, bias=False, activation=nn.ReLU(inplace=True))
        
        self.ese = ESEBlock_Conv(out_channels) if use_ese else nn.Identity()

    def forward(self, x):
        residual = x
        feats = [x]
        x = self.reduce_conv(x)
        for layer in self.residual_layers:
            x = layer(x)
            feats.append(x)
        concat_feat = torch.cat(feats, dim = 1)

        out = self.concat_conv(concat_feat)
        out = self.ese(out)

        if(self.identity != None):
            out += self.identity(residual)
        return out
    
class OSAStage(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, 
                 num_units: int, num_residuals: int = 5, num_ese_blocks: int = 0, kernel_size: int = 3, 
                 use_identity: bool = True, use_downsample: bool = False, use_depthwise: bool = False):
        """ Build a stage of VoV model.
        Args:
            in_channels: Number of channels in the input image
            internal_channels: Number of channels for osa residual.
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            num_residuals: Number of residual for osa unit.
            num_ese_blocks: Number of ese attention modeules in this stage.
            kernel_size: Size of the conv kernel
            use_identity: Whether to use identity for osa unit.
            use_downsample: Whether to use downsample.
            use_depthwise: Whether to use depthwise modules.
        """
        super(OSAStage, self).__init__()

        if use_downsample:
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=2, ceil_mode=True)
        else:
            self.pool = None

        units = []
        for ix in range(num_units):
            use_ese = False
            if num_ese_blocks > num_units:
                raise ValueError("Number of ESE blocks cannot exceed number of layers.")
            use_ese = (ix >= num_units - num_ese_blocks)
            if (use_depthwise) :
                units.append(DWsOSAUnit(in_channels, internal_channels, out_channels, kernel_size=kernel_size,
                                        num_residuals=num_residuals, use_identity=use_identity and ix > 0, use_ese=use_ese))
            else :
                units.append(OSAUnit(in_channels, internal_channels, out_channels, kernel_size=kernel_size,
                                     num_residuals=num_residuals, use_identity=use_identity and ix > 0, use_ese=use_ese))
            in_channels = out_channels
        self.osa_stage = nn.Sequential(*units)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        x = self.osa_stage(x)
        return x

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [128, 256,  1, 5, 0, False],
        [160, 512,  1, 5, 1, True],
        [192, 769,  4, 5, 1, True],
        [224, 1024, 3, 5, 1, True],
    ]
    input=torch.randn(3, 128, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        vov = OSAStage(input.size(1), internal_channels=opt[0], out_channels=opt[1], num_units=opt[2], num_residuals=opt[3], num_ese_blocks=opt[4],
                       use_downsample=opt[5])
        vov.to(device)
        vov.eval()
        
        # ori result
        out = vov(input)

        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(vov, input)
        summary(copy.deepcopy(vov).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(vov, input, f"vovnet_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")