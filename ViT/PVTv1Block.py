import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC, FFNBlock_Conv
from Transformer.utils import DropPath, LayerNorm, view_format
from Transformer.MultiHeadAttention import MultiHeadAttention

__all__ = ["PVTv1Unit", "PVTv1Stage"]
# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions (https://arxiv.org/abs/2102.12122)
# ICCV 2021

class SpatialReductionMultiHeadAttn(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0.,proj_drop_ratio: float = 0., sr_ratio: int = 1):
        super(SpatialReductionMultiHeadAttn, self).__init__(dim, num_heads, None, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
            self.w_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def internal_forword(self, inputs, in_format, out_format): # BCHW, BHWC, BNC
        inputs, pre_shape = view_format(inputs, in_format, "BNC")
        B, N, C = inputs.shape # (B, HW, C)
        inputs = self.pre_bn(inputs)

        if self.sr_ratio > 1.:
            q = self.w_q(inputs).reshape(B, N, self.num_heads, C // self.num_heads) \
                                .permute(0, 2, 1, 3)
            x_ = view_format(inputs, in_type="BNC", out_type="BCHW", shape=pre_shape)
            x_ = self.sr(x_)
            x_, _ = view_format(x_, in_type="BCHW", out_type="BNC")
            x_ = self.norm(x_)
            B_, N_, C_ = x_.shape # N_ = (H//sr_ratio)*(W//sr_ratio)

            kv = self.w_kv(x_).reshape(B_, N_, 2, self.num_heads, C_ // self.num_heads) \
                              .permute(2, 0, 3, 1, 4)
            q, (k, v) = q, kv.unbind(0)
        else :
            qkv = self.w_qkv(inputs).reshape(B, N, 3, self.num_heads, C // self.num_heads) \
                                    .permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0) # 3*[(B, num_heads, N, C//num_heads)]

        attn_out = self.attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, C)

        out = self.proj(attn_out)
        return view_format(out, "BNC", out_format, shape=pre_shape)

    def forward(self, inputs, in_format="BCHW", out_format="BNC"): # BCHW, BHWC, BNC
        return self.internal_forword(inputs, in_format=in_format, out_format=out_format)
    
class PVTv1Unit(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., drop_path_rate=0., 
                 sr_ratio=1., ffn_expansion = 4):
        super().__init__()

        self.attn = SpatialReductionMultiHeadAttn(
            dim =dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate, sr_ratio=sr_ratio)
        
        self.ffn_block = FFNBlock_FC(in_dims=dim, internal_dims=int(dim * ffn_expansion), drop_path_rate=drop_path_rate)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.attn(x, in_format="BCHW", out_format="BCHW")
        x = x + self.ffn_block(x, in_format="BCHW", out_format="BCHW")
        return x

class PVTv1Stage(nn.Module):
    def __init__(self, in_channels: int, num_units: int, num_heads: int, out_channels: Union[None, int] = None, stride: int = 1, 
                 drop_path_rates: Union[float, Tuple[float, float]] = 0., qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0., sr_ratio: float = 1., ffn_expansion: int = 4):
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
            units.append(PVTv1Unit(dim=in_channels, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop_ratio=attn_drop_ratio, drop_path_rate=block_drop_path, sr_ratio=sr_ratio,
                                ffn_expansion=ffn_expansion))

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

        pvt = PVTv1Unit(input.size(1), num_heads=opt[0], qk_scale=opt[1], sr_ratio=opt[2])
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