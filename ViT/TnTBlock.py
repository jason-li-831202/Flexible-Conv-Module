import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC, FFNBlock_Conv
from Transformer.utils import DropPath, LayerNorm, view_format
from Transformer.MultiHeadAttention import MultiHeadAttention

__all__ = ["TnTUnit"]
# Transformer in Transformer (https://arxiv.org/abs/2103.00112)
# NeurIPS 2021

class _SEBlock(nn.Module):
    def __init__(self, dims: int, internal_reduce: int = 16):
        """ Constructor
        Args:
            dims: input dimensionality.
            internal_reduce: the ratio for internal_dims.
        """
        super(_SEBlock, self).__init__()
        self.pre_bn = nn.LayerNorm(dims)
        self.fc_layers = nn.Sequential(OrderedDict([
            ('Squeeze', nn.Linear(in_features = dims, out_features = dims // internal_reduce, bias = False)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('Excitation', nn.Linear(in_features = dims // internal_reduce, out_features = dims, bias = False)),
            ('Tanh', nn.Tanh()),
        ]))

    def forward(self, x):
        y = x.mean(dim=1, keepdim=True) # B, 1, C
        y = self.fc_layers(self.pre_bn(y))
        return y * x
    
class InnerUnit(nn.Module):
    def __init__(self, inner_dim: int, inner_num_heads: int, 
                 attn_drop_ratio: float = 0.0, drop_path_rate: float = 0., ffn_expansion: int = 4, ):
        super().__init__()
        # Inners
        self.inner_attn = MultiHeadAttention( dim = inner_dim, num_heads=inner_num_heads, qkv_bias=False, qk_scale=None,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate)
        self.inner_ffn = FFNBlock_FC(in_dims=inner_dim, internal_dims=int(inner_dim * ffn_expansion), drop_path_rate=drop_path_rate)                    
    
    def forward(self, inner_tokens):
        B, C, N = inner_tokens.shape

        inner_tokens = inner_tokens + self.inner_attn(inner_tokens, in_format="BNC", out_format="BNC")
        inner_tokens = inner_tokens + self.inner_ffn(inner_tokens, in_format="BNC", out_format="BNC")
        return inner_tokens
    
class OuterUnit(nn.Module):
    def __init__(self, outer_dim: int, outer_num_heads: int, 
                 attn_drop_ratio: float = 0.0, drop_path_rate: float = 0., ffn_expansion: int = 4, use_se: bool = True):
        super().__init__()
        # Outer
        self.outer_attn = MultiHeadAttention( dim = outer_dim, num_heads=outer_num_heads, qkv_bias=False, qk_scale=None,
                                attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate)
        self.outer_ffn = FFNBlock_FC(in_dims=outer_dim, internal_dims=int(outer_dim * ffn_expansion))                    
    
        # SE
        self.se = _SEBlock(outer_dim, 4) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, outer_tokens):
        B, C, N = outer_tokens.shape

        outer_tokens = outer_tokens + self.outer_attn(outer_tokens, in_format="BNC", out_format="BNC")
        tmp_ = self.outer_ffn(outer_tokens, in_format="BNC", out_format="BNC")
        outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se(tmp_))

        return outer_tokens
    
class TnTUnit(nn.Module):
    def __init__(self, inner_dim: int, outer_dim: int, inner_num_heads: int, outer_num_heads: int, num_pixel: int,
                 attn_drop_ratio: float = 0.0, drop_path_rate: float = 0., ffn_expansion: int = 4, use_se: bool = True):
        """ 
        Args:
            inner_dim: 单个patch的维度大小(不包含pixel-level维度)
            outer_dim: patch的嵌入维度大小(也是实际输入数据的映射空间大小)
            inner_num_heads: Number of pxiel embeb per head  
            outer_num_heads: Number of patch embeb per head 
            num_pixel: patch下的(in_dim维度)元素对应pixel的比例 1:num_pixel, 即pixel个数(也是论文中指的patch2pixel分辨率)
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
            ffn_expansion: ratio of ffn hidden channels.
        """
        super().__init__()

        # Inners
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_unit = InnerUnit(inner_dim = inner_dim, inner_num_heads = inner_num_heads, 
                                        attn_drop_ratio = attn_drop_ratio, drop_path_rate = drop_path_rate, 
                                        ffn_expansion = ffn_expansion )

            self.proj = nn.Sequential(OrderedDict([
                    ("pre_bn",  nn.LayerNorm(num_pixel * inner_dim)),
                    ("linear",  nn.Linear(num_pixel * inner_dim, outer_dim, bias=True)),
                ])) 

        # Outer
        self.outer_unit = OuterUnit(outer_dim = outer_dim, outer_num_heads = outer_num_heads, 
                                    attn_drop_ratio = attn_drop_ratio, drop_path_rate = drop_path_rate, 
                                    ffn_expansion = ffn_expansion, use_se = use_se)
        
    def forward(self, pixel_embeb, patch_embeb):
        if self.has_inner:
            inner_tokens = self.inner_unit(pixel_embeb)
            B, N, C = patch_embeb.size()
            # 不在这里操作class_token
            patch_embeb[:,1:] = patch_embeb[:,1:] + self.proj(inner_tokens.reshape(B, N-1, -1)) # B, N, C
        outer_tokens = self.outer_unit(patch_embeb)
        return inner_tokens, outer_tokens


if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [8, 12, 4, 4, 0.1],
        [4, 12, 8, 8, 0.4],
        [4, 12, 4, 2, 0.3],
    ]
    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")
        inner_tokens = torch.randn(3, 128, 24).to(device)
        outer_tokens = torch.randn(inner_tokens.size(0), 
                                   inner_tokens.size(1)//opt[0]+1, 
                                   opt[0]*inner_tokens.size(2)).to(device)

        TnT = TnTUnit(  inner_dim=inner_tokens.size(2), outer_dim=outer_tokens.size(2), num_pixel=opt[0],
                        inner_num_heads=opt[2], outer_num_heads=opt[1], 
                        ffn_expansion=opt[3], drop_path_rate=opt[4])
        TnT.to(device)
        TnT.eval()
        
        # ori result
        out = TnT(inner_tokens, outer_tokens)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(TnT, input=(inner_tokens, outer_tokens))
        save_model_to_onnx(TnT, (inner_tokens, outer_tokens), "ori_model.onnx") 

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")