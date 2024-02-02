import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC, FFNBlock_Conv
from Transformer.utils import DropPath, LayerNorm, view_format
from Transformer.MultiHeadAttention import MultiHeadAttention


__all__ = ["EdgeViTUnit", "EdgeViTStage"]
# EdgeViTs: Competing Light-weight CNNs on Mobile Devices with Vision Transformers (https://arxiv.org/abs/2205.03436)
# ECCV 2022

class GlobalMultiHeadAttn(MultiHeadAttention):
    def __init__(self, dim, num_heads=8, 
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0.,proj_drop_ratio=0.,  sr_ratio=1):
        super(GlobalMultiHeadAttn, self).__init__(dim, num_heads, None, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)
        self.sr = sr_ratio

        if self.sr > 1:
            self.sampler = nn.AvgPool2d(kernel_size=1, stride=sr_ratio)
            self.localProp = nn.ConvTranspose2d(dim, dim, kernel_size=sr_ratio, 
                                                stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sampler = nn.Identity()
            self.localProp = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, inputs, in_format="BCHW", out_format="BNC"): # BCHW, BHWC, BNC
        inputs, pre_shape = view_format(inputs, in_format, "BNC")
        B, N, C = inputs.shape 
        inputs = self.pre_bn(inputs)

        if self.sr > 1.:
            inputs = view_format(inputs, "BNC", "BCHW", shape=pre_shape)
            inputs = self.sampler(inputs)
            inputs, _ = view_format(inputs, "BCHW", "BNC")
            B, N, C = inputs.shape 

        qkv = self.w_qkv(inputs).reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='trunc')) \
                                .permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # 3*[(B, num_heads, N, C//num_heads)]
        
        attn_out = self.attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, C)

        if self.sr > 1:
            _shape = (*pre_shape[:2], pre_shape[-2]//self.sr, pre_shape[-1]//self.sr)
            attn_out = view_format(attn_out, "BNC", "BCHW", shape=_shape)
            attn_out = self.localProp(attn_out)
            attn_out, _ = view_format(attn_out, "BCHW", "BNC")
            attn_out = self.norm(attn_out)

        out = self.proj(attn_out)
        return view_format(out, "BNC", out_format, shape=pre_shape)

class EdgeViTUnit(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., drop_path_rate=0., 
                 sr_ratio=1., ffn_expansion = 4):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.attn = GlobalMultiHeadAttn(
            dim = dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate, sr_ratio=sr_ratio)
        
        self.ffn_block = FFNBlock_FC(in_dims=dim, internal_dims=int(dim * ffn_expansion), drop_path_rate=drop_path_rate)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape

        x = x + self.attn(x, in_format="BCHW", out_format="BCHW")
        x = x + self.ffn_block(x, in_format="BCHW", out_format="BCHW")
        return x

class LocalAggregation(nn.Module):
    def __init__(self, dim, ffn_expansion=4., drop_path_rate=0.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.pre_norm = nn.BatchNorm2d(dim)
        self.bottle_neck = nn.Sequential(OrderedDict([
            ("pw1_conv", nn.Conv2d(dim, dim, kernel_size=1)),
            ("dw_conv", nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)),
            ("pw2_conv", nn.Conv2d(dim, dim, kernel_size=1))
        ]))
        self.ffn_block = FFNBlock_Conv(in_channels=dim, internal_channels=int(dim * ffn_expansion))
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)

        x = x + self.drop_path(self.bottle_neck(self.pre_norm(x)))
        x = x + self.drop_path(self.ffn_block(x, in_format="BCHW", out_format="BCHW"))
        return x

class EdgeViTStage(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0., drop_path_rate: float = 0., 
                 sr_ratio: float = 1., ffn_expansion: int = 4):
        """ Build a stage of local-global-local bottleneck.
        Args:
        """
        super().__init__()
        
        if sr_ratio > 1:
            self.LocalAgg = LocalAggregation(dim=dim, ffn_expansion=ffn_expansion, drop_path_rate=drop_path_rate)
        else:
            self.LocalAgg = nn.Identity()
        
        self.SelfAttn = EdgeViTUnit(dim = dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop_ratio=attn_drop_ratio, drop_path_rate=drop_path_rate, sr_ratio=sr_ratio,
                                    ffn_expansion=ffn_expansion)
    def forward(self, x):
        x = self.LocalAgg(x)
        x = self.SelfAttn(x)
        return x
    
if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [8, 1.0, 2],
        [4, 0.5, 1],
        [8, 2.5, 0],
    ]
    input=torch.randn(24, 64, 32, 32).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        convNeXt = EdgeViTStage(input.size(1), num_heads=opt[0], qk_scale=opt[1], sr_ratio=opt[2])
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



