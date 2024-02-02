import torch, sys
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from typing import Union, Tuple

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN
from Transformer.FFNModule import FFNBlock_Conv
from Transformer.utils import DropPath 

__all__ = ["RepLKBlock", "RepLKUnit", "RepLKStage"]
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# CVPR 2022

class _RepaLKConv(FusedBase):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1,
                 small_kernel = None, fused = False, use_nonlinear=True, use_bias=False):
        super(_RepaLKConv, self).__init__(in_channels, out_channels, kernel_size, groups, stride)
        self.dilation = dilation
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        self.padding =  (dilation*(self.kernel_size[0]-1)//2, dilation*(self.kernel_size[1]-1)//2)
        self.activation = nn.ReLU() if use_nonlinear else nn.Identity()
        self.use_bias = use_bias
        self.fused = fused

        if fused:
            self.lkb_reparam = self._fuse_branch()
        else:
            self.large_conv = ConvBN(in_channels, out_channels, kernel_size=self.kernel_size, 
                                    stride=stride, dilation=dilation, padding=self.padding, groups=groups, bias=use_bias)
            if small_kernel is not None:
                if not isinstance(small_kernel, tuple):
                    from itertools import repeat
                    small_kernel = tuple(repeat(small_kernel, 2))

                smallK_padding = (dilation*(small_kernel[0]-1)//2, dilation*(small_kernel[1]-1)//2)
                assert small_kernel[0] <= self.kernel_size[0], 'The small_kernel size w for re-param cannot be larger than the large kernel!'
                assert small_kernel[1] <= self.kernel_size[1], 'The small_kernel size h for re-param cannot be larger than the large kernel!'
                self.small_conv = ConvBN(in_channels, out_channels, kernel_size=small_kernel, 
                                        stride=stride, dilation=dilation, padding=smallK_padding, groups=groups, bias=use_bias)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            return self.activation(self.lkb_reparam(inputs))

        out = self.large_conv(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        return self.activation(out)

    def merge_kernel(self):
        if (self.fused): return 0
        self.fused = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.lkb_reparam = self._fuse_branch()

        self.lkb_reparam.weight.data = kernel
        if (self.use_bias) :
            self.lkb_reparam.bias.data = bias
        self.del_ori_bracnch()

    def _fuse_branch(self) :
        return nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel,
                            kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                            padding_mode='zeros', stride=self.stride,
                            groups=self.groups, bias=self.use_bias)


    def _get_equivalent_kernel_bias(self):
        kernel_large, bias_large = self._fuse_conv_bn_module(self.large_conv)

        if hasattr(self, 'small_conv'):
            kernel_small, bias_small = self._fuse_conv_bn_module(self.small_conv)

        return self._fuse_addbranch( (kernel_large, self._pad_kernel(kernel_small)), 
                                      (bias_large, bias_small))

    def del_ori_bracnch(self):
        # 消除梯度更新
        [para.detach_() for para in self.parameters()]

        # del no use branch 
        self.__delattr__('large_conv')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class RepLKBlock(nn.Module):
    def __init__(self, channels: int, dw_channels: int, block_lk_size: Union[int, Tuple[int, int]], block_sk_size: Union[int, Tuple[int, int]], 
                 drop_path_rate: float, fused: bool = False):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            dw_channels: internal channels for dw group.
            block_lk_size: large kernel size for RepaLKConv Module.
            block_sk_size: small kernel size for RepaLKConv Module.
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        """
        super().__init__()
        self.lkb_pre_bn = nn.BatchNorm2d(channels)
        self.lkb_pw1 = nn.Sequential(OrderedDict([  
            ('conv', nn.Conv2d(channels, dw_channels,
                               kernel_size = 1, stride = 1, padding = 0, groups = 1, bias=False)),
            ('relu', nn.ReLU()),

        ]))         
        self.lkb_dw = _RepaLKConv(in_channels=dw_channels, out_channels=dw_channels, kernel_size=block_lk_size,
                                        stride=1, dilation=1, groups=dw_channels, small_kernel=block_sk_size, fused=fused)
        self.lkb_pw2 = ConvBN(dw_channels, channels, kernel_size=1, 
                    stride=1, dilation=1, padding=0, groups=1, bias=False) 
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        print('RepLK Block, identity = ', self.drop_path)

    def forward(self, x):
        out = self.lkb_pre_bn(x)
        out = self.lkb_pw1(out)
        out = self.lkb_dw(out)
        out = self.lkb_pw2(out)
        return self.drop_path(out)

    def merge_kernel(self):
        if (not self.lkb_dw.fused):
            self.lkb_dw.merge_kernel()

# Assume all RepLK Blocks within a stage share the same lk_size. You may tune it on your own model.
class RepLKUnit(nn.Module):
    def __init__(self, channels: int, block_lk_size: int, block_sk_size: int, 
                 drop_path_rate: float, dw_ratio: float = 1, ffn_ratio: float = 4,
                 fused: bool = False):
        """ Initialization of the class.
        Args:
            channels: input channel dimensionality.
            block_lk_size: large kernel size for RepaLKConv Module.
            block_sk_size: small kernel size for RepaLKConv Module.
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
            dw_ratio: Dw group for RepLK block by rate of input channel.
            ffn_ratio: Internal channels for ConvFFN block by rate of input channel.
        """
        super().__init__()

        self.replk_block = RepLKBlock(channels = channels, dw_channels = int(channels * dw_ratio), 
                                    block_lk_size = block_lk_size, block_sk_size = block_sk_size, 
                                    drop_path_rate = drop_path_rate , fused = fused)
        self.ffn_block = FFNBlock_Conv(in_channels = channels, internal_channels = int(channels * ffn_ratio), out_channels = channels,
                                        drop_path_rate = drop_path_rate)
 
    def forward(self, x):
        x = x + self.replk_block(x)
        x = x + self.ffn_block(x, in_format="BCHW", out_format="BCHW")
        return x

    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, RepLKBlock):
                module.merge_kernel()

class RepLKStage(nn.Module):
    def __init__(self, channels: int, num_units: int, stage_lk_size: int, stage_sk_size: int, 
                 drop_path_rates: Union[float, Tuple[float, float]], dw_ratio: float = 1, ffn_ratio: float = 4,
                 fused: bool = False):
        """ Build a stage of RepLK model.
        Args:
            channels: input channel dimensionality.
            num_units: Number of unit modeules in this stage.
            block_lk_size: large kernel size for RepaLKConv Module.
            block_sk_size: small kernel size for RepaLKConv Module.
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
            dw_ratio: Dw group for RepLK block by rate of input channel.
            ffn_ratio: Internal channels for ConvFFN block by rate of input channel.
        """
        super().__init__()
        if isinstance(drop_path_rates, list) :
            assert num_units==len(drop_path_rates), "num of drop path list must be consistent with num_units"
        
        units = []
        for i in range(num_units):
            block_drop_path = drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates
            units.append(RepLKUnit(channels = channels, block_lk_size = stage_lk_size, block_sk_size = stage_sk_size,
                         drop_path_rate = block_drop_path, dw_ratio = dw_ratio, ffn_ratio = ffn_ratio, fused=fused ))
        self.rep_lk_stage = nn.ModuleList(units)

    def forward(self, x):
        for unit in self.rep_lk_stage:
            x = unit(x)
        return x

    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, RepLKUnit):
                module.merge_kernel()

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = [
        [(3, 3), (1, 1), 0.0],
        [(7, 7), (5, 1), 0.4],
        [(13, 13), (3, 5), 0.1],
        [31, (7, 7), 0.2],
    ]
    input=torch.randn(2, 16, 112, 112).to(device)

    print(f"--- Comparing blocks ---")
    for opt in options:
        print(f"Comparing for {opt} ...")

        rep_lk = RepLKUnit(input.size(1), block_lk_size=opt[0], block_sk_size=opt[1], drop_path_rate=opt[2])
        rep_lk.to(device)
        rep_lk.eval()

        # ori result
        out = rep_lk(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(rep_lk, input)
        summary(copy.deepcopy(rep_lk).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(rep_lk, input, "ori_model.onnx") 

        # fuse result
        rep_lk.merge_kernel()
        out2 = rep_lk(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(rep_lk, input)
        summary(copy.deepcopy(rep_lk).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(rep_lk, input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")