import torch, sys
from torch import nn
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC
from Transformer.utils import DropPath, LayerNorm

__all__ = ["ConvNeXtUnit", "ConvNeXtStage"]
# A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545)
# 	CVPR 2022

class ConvNeXtUnit(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> FFN; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> FFN; Permute back
    We use (2) as we find it slightly faster in PyTorch
    """
    def __init__(self, dims: int, ffn_expansion: int = 4, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6):
        """ Constructor
        Args:
            dims: Number of input channels.
            ffn_expansion: ratio of ffn hidden channels.
            drop_path_rate: Stochastic depth rate. Default: 0.0
            layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()

        self.dw_conv = nn.Conv2d(dims, dims, kernel_size=7, padding=3, groups=dims) 

        self.ffn_block = FFNBlock_FC(dims, ffn_expansion * dims, drop_path_rate=drop_path_rate)
        self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dims)), 
                                            requires_grad=True) if layer_scale_init_value > 0 else None
        
    def forward(self, x):
        B, C, H, W  = x.shape
        
        input = x
        x = self.dw_conv(x)

        x = self.ffn_block(x, in_format="BCHW", out_format = "BCHW")
        x = self.ffn_gamma.view(C, 1, 1) * x if self.ffn_gamma is not None else x
        x = input + x

        return x
    
class ConvNeXtStage(nn.Module):
    def __init__(self, in_channels: int, num_units: int, out_channels: Union[None, int] = None, stride: int = 1,
                 drop_path_rates: Union[float, Tuple[float, float]] = 0., layer_scale_init_value: float=1e-6):
        """ Build a stage of ConvNeXt model.
        Args:
            in_channels: input channel dimensionality.
            num_units: Number of unit modeules in this stage.
            out_channels: Number of channels produced by the convolution.
            stride: Stride of the convolution.
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
            layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        if isinstance(drop_path_rates, list) :
            assert num_units==len(drop_path_rates), "num of drop path list must be consistent with num_units"

        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride, bias=True))
            in_channels = out_channels
        else:
            self.downsample = nn.Identity()

        units = []
        for i in range(num_units):
            block_drop_path = drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates
            units.append(ConvNeXtUnit(dims=in_channels, drop_path_rate=block_drop_path, 
                         layer_scale_init_value=layer_scale_init_value))
        self.hor_stage = nn.Sequential(*units)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.downsample(x)
        x = self.hor_stage(x)
        return x
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [3, 0.0],
        [3, 0.5],
        [9, 0.2],
    ]
    input=torch.randn(3, 96, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        convNeXt = ConvNeXtStage(input.size(1), num_units=opt[0], drop_path_rates=opt[1], stride=3)
        convNeXt.to(device)
        convNeXt.eval()
        
        # ori result
        out = convNeXt(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(convNeXt, input)
        summary(copy.deepcopy(convNeXt).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(convNeXt, input, "ori_model.onnx") 

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")