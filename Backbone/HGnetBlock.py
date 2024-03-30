import torch, sys
from torch import nn
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Transformer.utils import DropPath
from Attention.SEModule import ESEBlock_Conv

__all__ = ["HGUnit", "HGStage"]
# High Performance GPU Net ( https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/models/ImageNet1k/PP-HGNet.md )
# paddlepaddle

class _LightConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 stride: int = 1, dilation: int = 1, activation=nn.ReLU()):
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.padding =  (dilation*(kernel_size[0]-1)//2, dilation*(kernel_size[1]-1)//2)

        self.pw_conv = ConvBN(in_channels, out_channels, kernel_size=1, stride=1)
        self.dw_conv = ConvBN(out_channels, out_channels, kernel_size=kernel_size, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=out_channels, activation=activation)

    def forward(self, x):
        return self.dw_conv(self.pw_conv(x))
    

class HGUnit(nn.Module):
    def __init__(self, in_channels, internal_channels, out_channels, kernel_size: Union[int, Tuple[int, int]] = 3, 
                 num_residuals: int = 6, light_block: bool = False, use_ese: bool = True, drop_path_rate: float = 0.):
        super().__init__()
        block = _LightConv2d if light_block else ConvBN

        self.levels = nn.ModuleList(block(in_channels if i == 0 else internal_channels, internal_channels, kernel_size=kernel_size, activation=nn.ReLU())\
                                for i in range(num_residuals))
        
        # feature aggregation
        total_channels = in_channels + num_residuals * internal_channels
        if use_ese :
            self.concat_conv = nn.Sequential(OrderedDict([
                ('Squeeze', ConvBN(total_channels, out_channels, kernel_size=1, stride=1, activation=nn.ReLU())),
                ('Excitation',  ESEBlock_Conv(out_channels) ),
            ]))
        else :
            self.concat_conv = nn.Sequential(OrderedDict([
                ('Squeeze', ConvBN(total_channels, out_channels//2, kernel_size=1, stride=1, activation=nn.ReLU())),
                ('Excitation', ConvBN(out_channels // 2, out_channels, kernel_size=1, stride=1, activation=nn.ReLU()) ),
            ]))
        
        self.identity = nn.Identity() if in_channels == out_channels else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        residual = 0 if (self.identity == None) else self.identity(x)

        feats = [x]
        feats.extend(m(feats[-1]) for m in self.levels)

        concat_feat = torch.cat(feats, dim = 1)
        out = self.concat_conv(concat_feat)
        return self.drop_path(out) + residual
    

class HGStage(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, num_units: int, kernel_size: Union[float, Tuple[float, float]]  = (3, 3), 
                 drop_path_rates: Union[float, Tuple[float, float]]= 0., num_residuals: int = 6, use_light: bool = False, use_ese: bool = False) :
        """ Build a stage of HGnet model.
        Args:
            in_channels: Number of channels in the input image
            internal_channels: Number of channels for osa residual.
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            kernel_size: Size of the conv kernel
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
            num_residuals: Number of residual for hg unit.
            use_light: use light convolution.
            use_ese: use ese attention modeules.
        """
        super(HGStage, self).__init__()
        if isinstance(drop_path_rates, list) :
            assert num_units==len(drop_path_rates), "num of drop path list must be consistent with num_units"
        
        self.downsample = ConvBN(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)

        units = []
        for i in range(num_units):
            block_drop_path = drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates
            units.append(HGUnit(in_channels, internal_channels, out_channels, kernel_size=kernel_size, 
                                num_residuals=num_residuals, light_block=use_light, use_ese=use_ese, drop_path_rate=block_drop_path))
            in_channels = out_channels
        self.hgnet_stage = nn.Sequential(*units)

    def forward(self, x):
        x = self.downsample(x)
        return self.hgnet_stage(x)
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage_config =[
        [32,  256,  1, 3, 3, True],
        [64,  512,  2, (5, 5), 3, True],
        [128, 1028, 1, 5, 3, True],
    ]
    input=torch.randn(1, 64, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for i, opt in enumerate(stage_config):
        torch.cuda.empty_cache() 
        print(f"Comparing for {opt} ...")

        hgnet = HGStage(input.size(1), internal_channels=opt[0], out_channels=opt[1], num_units=opt[2], 
                        kernel_size=opt[3], num_residuals=opt[4], use_light=opt[5])
        hgnet.to(device)
        hgnet.eval()
        
        # ori result
        out = hgnet(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(hgnet, input)
        summary(copy.deepcopy(hgnet).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(hgnet.to("cpu"), input.to("cpu"), f"hg_stage{i}.onnx") 
        input = out

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")
