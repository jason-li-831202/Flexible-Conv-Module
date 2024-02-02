import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from collections import OrderedDict

from Transformer.utils import LayerNorm, view_format
warnings.simplefilter('ignore') 

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, scale: float, attn_dropout: float = 0.1):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim = -1)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        output = torch.matmul(attn, v)
        return output
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,  dim: int, 
                        num_heads: int = 8, 
                        head_dim: Union[None, int] = None, 
                        qkv_bias: bool = False, 
                        qk_scale: Union[None, float] = None, 
                        attn_drop_ratio: float = 0.0, 
                        proj_drop_ratio: float = 0.0):
        """ 
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: Temperature (multiply) of softmax activation
            attn_drop_ratio: Dropout ratio of attention weight. Default: 0.0
            proj_drop_ratio: Dropout ratio of output. Default: 0.0
        """

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = head_dim or dim // num_heads 
        self.scale = qk_scale or head_dim ** -0.5
        self.num_heads = num_heads
        self.attn_dim = head_dim * self.num_heads  

        self.pre_bn = LayerNorm(dim, eps=1e-6)
        self.w_qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias)
        self.attention = ScaledDotProductAttention(scale=self.scale, attn_dropout=attn_drop_ratio)

        use_proj = not (num_heads == 1 and self.attn_dim == dim)
        self.proj = nn.Sequential(OrderedDict([
                    ("linear", nn.Linear(self.attn_dim, dim, bias=qkv_bias)),
                    ("dropout", nn.Dropout(proj_drop_ratio))
            ]))  if use_proj else nn.Identity()

    def internal_forword(self, inputs, in_format, out_format):
        with warnings.catch_warnings():
            if (in_format != "BNC") :
                inputs, shape = view_format(inputs, in_format, "BNC")
            else :
                shape = None  
            B, N, C = inputs.shape 
            inputs = self.pre_bn(inputs)

            qkv = self.w_qkv(inputs)
            C_ = qkv.shape[2] // 3
            qkv = qkv.reshape(B, N, 3, self.num_heads, C_ // self.num_heads) \
                            .permute(2, 0, 3, 1, 4) # (B, N, 3*attn_dim) -> (B, N, 3, num_heads, attn_dim//num_heads) -> (3, B, num_heads, N, attn_dim//num_heads)
            
            
            q, k, v = qkv[0], qkv[1], qkv[2] # 3*[(B, num_heads, N, attn_dim//num_heads)]
            attn_out = self.attention(q, k, v) 
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, C_) # (B, num_heads, N, attn_dim//num_heads) -> (B, N, attn_dim)

            out = self.proj(attn_out)
            return view_format(out, "BNC", out_format, shape=shape)

    def forward(self, inputs, in_format="BCHW", out_format="BNC"): # BCHW, BHWC, BNC
        return self.internal_forword(inputs, in_format=in_format, out_format=out_format)
