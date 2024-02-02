import torch, sys, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC, FFNBlock_Conv
from Transformer.utils import DropPath, LayerNorm, view_format
from Transformer.MultiHeadAttention import MultiHeadAttention


__all__ = ["PVTv2Unit", "PVTv2Stage"]
# PVT v2: Improved Baselines with Pyramid Vision Transformer (https://arxiv.org/abs/2106.13797)
# CVMJ 2022

class SpatialReductionMultiHeadAttn(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0.,proj_drop_ratio: float = 0., sr_ratio: int = 1, linear: bool = False):
        super(SpatialReductionMultiHeadAttn, self).__init__(dim, num_heads, None, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.__delattr__('w_qkv')

        self.sr_ratio = sr_ratio
        self.linear = linear
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            # self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _adaptiveAvgPool(self, x, in_size, out_size=[7, 7]):
        in_shape = np.array(in_size)
        out_shape = np.array(out_size)
        stride_ = np.floor(in_shape/out_shape).astype(np.int32)
        kernel_ = in_shape -(out_shape-1)*stride_
        return F.avg_pool2d(x, list(kernel_), list(stride_))

    def forward(self, inputs, in_format="BCHW", out_format="BNC"): # BCHW, BHWC, BNC
        inputs, pre_shape = view_format(inputs, in_format, "BNC")
        B, N, C = inputs.shape # (B, HW, C)
        inputs = self.pre_bn(inputs)

        q = self.w_q(inputs).reshape(B, N, self.num_heads, C // self.num_heads) \
                            .permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1.:
                x_ = view_format(inputs, in_type="BNC", out_type="BCHW", shape=pre_shape)
                x_ = self.sr(x_)
                x_, _ = view_format(x_, in_type="BCHW", out_type="BNC")
                x_ = self.norm(x_)
                B_, N_, C_ = x_.shape # N_ = (H//sr_ratio)*(W//sr_ratio)
            else :
                x_ = inputs
                B_, N_, C_ = x_.shape # N_ = (H*W)
        else:
            x_ = view_format(inputs, in_type="BNC", out_type="BCHW", shape=pre_shape)
            x_ = self._adaptiveAvgPool(x_, in_size=pre_shape[2:], out_size=[7, 7])
            x_ = self.sr(x_)
            # x_ = self.sr(self.pool(x_))
            x_, _ = view_format(x_, in_type="BCHW", out_type="BNC")
            x_ = self.norm(x_)
            x_ = self.act(x_)
            B_, N_, C_ = x_.shape # N_ = 7*7
        kv = self.w_kv(x_).reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads) \
                          .permute(2, 0, 3, 1, 4)
        q, (k, v) = q, kv.unbind(0)
            
        attn_out = self.attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, C)

        out = self.proj(attn_out)
        return view_format(out, "BNC", out_format, shape=pre_shape)

class FFNDwBlock_FC(FFNBlock_FC):
    def __init__(self, in_dims: int, internal_dims: int = None, out_dims: int = None, 
                 drop_path_rate: float = 0., linear: bool = False):
        super(FFNDwBlock_FC, self).__init__(in_dims, internal_dims, out_dims, True, drop_path_rate)
        
        self.dw_conv = nn.Conv2d(internal_dims, internal_dims, kernel_size=3, stride=1, padding=1, 
                                 groups=internal_dims, bias=True) 
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, in_format="BCHW", out_format = "BCHW"): # (B, C, H, W)
        x, shape = view_format(x, in_format, "BNC")
        out = self.ffn_pre_bn(x)
        out = self.ffn_fc1(out)

        out = self.relu(out) if self.linear else out
        out = view_format(out, "BNC", "BCHW", shape=shape)
        out = self.dw_conv(out)
        out, _ = view_format(out, "BCHW", "BNC")
    
        out = self.activation(out)
        out = self.ffn_fc2(out)
        return view_format(self.drop_path(out), "BNC", out_format, shape=shape)
    
class PVTv2Unit(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., drop_path_rate=0., 
                 sr_ratio=1., linear=False, ffn_expansion = 4):
        super().__init__()

        self.attn = SpatialReductionMultiHeadAttn(
            dim =dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate, sr_ratio=sr_ratio, linear=linear)
        
        self.ffn_block = FFNDwBlock_FC(in_dims=dim, internal_dims=int(dim * ffn_expansion), 
                                       drop_path_rate=drop_path_rate, linear=linear)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.attn(x, in_format="BCHW", out_format="BCHW")
        x = x + self.ffn_block(x, in_format="BCHW", out_format="BCHW")
        return x

class PVTv2Stage(nn.Module):
    def __init__(self, in_channels: int, num_units: int, num_heads: int, out_channels: Union[None, int] = None, stride: int = 1, 
                 drop_path_rates: Union[float, Tuple[float, float]] = 0., qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0., sr_ratio: float = 1., linear: bool = False, ffn_expansion: int = 4):
        """ Build a stage of PVT model.
        Args:
            in_channels: input channel dimensionality.
            num_units: Number of unit modeules in this stage.
            num_heads: Number of attention heads.
            out_channels: Number of channels produced by the convolution.
            stride: Stride of the convolution.
            drop_path_rates: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: Temperature (multiply) of softmax activation
            attn_drop_ratio: Dropout ratio of attention weight. Default: 0.0
            sr_ratio: ratio of spatial reduction.
            linear: whether to use linear spatial reduction attention.
            ffn_expansion: ratio of ffn hidden channels.
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
            units.append(PVTv2Unit(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, drop_path_rate=block_drop_path, sr_ratio=sr_ratio,
                                linear=linear, ffn_expansion=ffn_expansion))

        self.pvt_stage = nn.Sequential(*units)
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
        x = self.pvt_stage(x)
        return x
    
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [8, 1.0, 2],
        [4, 0.5, 1],
        [8, 2.5, 8],
    ]
    input=torch.randn(24, 8, 32, 32).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        pvt = PVTv2Unit(input.size(1), num_heads=opt[0], qk_scale=opt[1], sr_ratio=opt[2])
        pvt.to(device)
        pvt.eval()
        
        # ori result
        out = pvt(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(pvt, input)
        summary(copy.deepcopy(pvt).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(pvt, input, "ori_model.onnx") 

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")