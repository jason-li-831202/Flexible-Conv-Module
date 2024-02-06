import torch, sys, math
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from einops import rearrange
from collections import OrderedDict

sys.path.append("..")
from Conv.DwsConv import DepthwiseSeparableConv2d
from Transformer.FFNModule import FFNBlock_Conv
from Transformer.MultiHeadAttention import MultiHeadAttention
from Transformer.utils import DropPath, LayerNorm, view_format

__all__ = ["STViTUnit", "STViTStage"]
# Vision Transformer with Super Token Sampling (https://arxiv.org/abs/2211.11167)
# CVPR 2023

class _FFNBlock_Conv(FFNBlock_Conv):
    def __init__(self, in_channels: int, internal_channels: int=None, out_channels: int=None, 
                 use_nonlinear: bool=True, drop_path_rate: float=0.):
        super(_FFNBlock_Conv, self).__init__(in_channels, internal_channels, out_channels, use_nonlinear, drop_path_rate)
        hidden_features = internal_channels or in_channels

        self.dw_conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)

    def forward(self, x, in_format="BCHW", out_format = "BCHW"): # (B, C, H, W)
        x = view_format(x, in_format, "BCHW")
        out = self.ffn_pre_bn(x)
        out = self.ffn_pw1(out)
        out = self.activation(out)
        out = out + self.dw_conv(out)
        out = self.ffn_pw2(out)
        return view_format(self.drop_path(out), "BCHW", out_format)
    
class _Unfold(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        B, C, H, W = x.shape
        x = F.conv2d(x.reshape(B*C, 1, H, W), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(B, C*9, H*W)

class _Fold(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
    def forward(self, x):
        B, C, H, W = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x
    
class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, 
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., drop_path_rate=0.):
        super().__init__()
        stoken_size = _pair(stoken_size)
        
        self.n_iter = n_iter
        self.ph, self.pw = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention  
        
        self.scale = dim ** - 0.5
        
        self.unfold = _Unfold(kernel_size=3)
        self.fold = _Fold(kernel_size=3)
        self.pre_bn = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        if refine:
            if refine_attention:
                self.stoken_refine = MultiHeadAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
                    DepthwiseSeparableConv2d(dim, dim, kernel_size=5, stride=1, padding=2,
                                             bias=True, use_bn=False, use_nonlinear=False)
                )

    def _interpolate(self, x):
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / self.ph) * self.ph, math.ceil(W / self.pw) * self.pw
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x, (B, C, H, W)
    
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        x, pre_shape = self._interpolate(x)
        B, C, H, W = x.shape

        H_, W_ = H//self.ph, W//self.pw
        stoken_features = F.adaptive_avg_pool2d(x, (H_, W_)) # (B, C, H_, W_)

        pixel_features = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw) c', ph=self.ph, pw=self.pw)
        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features) # (B, C*9, H_*W_)

                stoken_features = stoken_features.transpose(1, 2).reshape(B, H_*W_, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale # (B, H_*W_, ph*pw, 9)
                affinity_matrix = affinity_matrix.softmax(-1) # (B, H_*W_, ph*pw, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, H_, W_)

                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, H_*W_, C, 9)
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, H_, W_)).reshape(B, C, H_, W_)            
                    stoken_features = stoken_features/(affinity_matrix_sum + 1e-12) # (B, C, H_, W_)
        
        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix # (B, H_*W_, C, 9)
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B*C, 9, H_, W_)).reshape(B, C, H_, W_)            
        stoken_features = stoken_features/(affinity_matrix_sum.detach() + 1e-12) # (B, C, H_, W_)

        stoken_features = self.direct_forward(stoken_features)

        stoken_features = self.unfold(stoken_features) # (B, C, H_, W_) -> (B, C*9, H_*W_)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, H_*W_, C, 9) # (B, C*9, H_*W_) -> (B, H_*W_, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2) # (B, H_*W_, C, ph*pw)

        pixel_features = rearrange(pixel_features, 'b (h w) c (ph pw) -> b c (h ph) (w pw)', h=H_, w=W_, ph=self.ph, pw=self.pw)

        if x.shape != pre_shape:
            pixel_features = F.interpolate(pixel_features, size=(pre_shape[2], pre_shape[3]), mode="bilinear", align_corners=False)

        return pixel_features
    
    def direct_forward(self, x):
        B, C, H, W = x.shape
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(x, in_format="BCHW", out_format="BCHW")
            else:
                stoken_features = self.stoken_refine(x)
        else :
            stoken_features = x
        return stoken_features
        
    def forward(self, x):
        x = self.pre_bn(x)
        if self.ph > 1 or self.pw > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)
        
class STViTUnit(nn.Module):
    def __init__(self, dim: int, num_iters: int, patch_size: Union[int, Tuple[int, int]], num_heads: int = 1, ffn_expansion: int = 4, 
                 qkv_bias: bool = False, qk_scale: Union[None, float] = None, 
                 attn_drop_ratio: float = 0., drop_path_rate: float = 0, layer_scale_init_value: float = 0.):
        super().__init__()
                        
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
                                        

        self.attn = StokenAttention(
                dim, stoken_size=patch_size, n_iter=num_iters, num_heads=num_heads, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio)   
        self.attn_gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                       requires_grad=True) if layer_scale_init_value > 0 else None
        

        self.ffn_block = _FFNBlock_Conv(in_channels=dim, internal_channels=int(dim * ffn_expansion))
        self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape

        input = x
        x = self.attn(x)
        x = self.attn_gamma * x if self.attn_gamma is not None else x
        x = input + self.drop_path(x)

        input = x
        x = self.ffn_block(x, in_format="BCHW", out_format="BCHW")
        x = self.ffn_gamma * x if self.ffn_gamma is not None else x
        x = input + self.drop_path(x)

        return x
    
class STViTStage(nn.Module):
    def __init__(self, in_channels: int, num_units: int, num_iters: int, patch_size: Union[int, Tuple[int, int]], num_heads: int, out_channels: Union[None, int] = None, 
                 stride: int = 1, attn_drop_ratio: float = 0., drop_path_rates: Union[float, Tuple[float, float]] = 0., 
                 qkv_bias: bool = False, qk_scale: Union[None, float] = None, layer_scale_init_value: float=0., ffn_expansion: int = 4):
        """ Build a stage of STViTStage model.
        Args:
            in_channels: input channel dimensionality.
            num_units: Number of unit modeules in this stage.
            num_iters: Number of iter for super token sampling.
            patch_size: Size of patches. image_size must be divisible by patch_size.
            num_heads: Number of attention heads.
            out_channels: Number of channels produced by the convolution.
            stride: Stride of the convolution.
            attn_drop_ratio: Dropout ratio of attention weight. Default: 0.0
            drop_path_rates: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
            qkv_bias: If True, add a learnable bias to query, key, value. Default: True
            qk_scale: Temperature (multiply) of softmax activation
            layer_scale_init_value: Init value for Layer Scale. Default: 0.
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
            units.append(STViTUnit(dim=in_channels, num_iters=num_iters, patch_size=patch_size, num_heads=num_heads, ffn_expansion=ffn_expansion, 
                 qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, 
                 drop_path_rate=block_drop_path, layer_scale_init_value=layer_scale_init_value))
        self.stvit_stage = nn.ModuleList(units)
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
        for unit in self.stvit_stage:
            x = unit(x)
        return x
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [1, 8, 8, 1.0],
        [1, 4, 4, 0.5],
        [1, 4, 8, 2.5],
    ]
    input=torch.randn(3, 8, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        stvit = STViTUnit(input.size(1), num_iters=opt[0], patch_size=opt[1], num_heads=opt[2], qk_scale=opt[3])
        stvit.to(device)
        stvit.eval()
        
        # ori result
        out = stvit(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(stvit, input)
        summary(copy.deepcopy(stvit).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(stvit, input, "ori_model.onnx") 

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")