import torch, sys, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Union, Tuple
from einops import rearrange
from collections import OrderedDict

sys.path.append("..")
from Transformer.FFNModule import FFNBlock_FC
from Transformer.MultiHeadAttention import MultiHeadAttention
from Conv.Basic.common import ConvBN

__all__ = ["MobileViTUnit", "MobileViTStage"]
# MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer (https://arxiv.org/abs/2110.02178)
# ICLR 2021


class MobileViTUnit(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, dim_head: int, 
                 attn_drop_ratio: float = 0.0, drop_path_rate: float = 0., ffn_expansion: int = 4, ):
        super().__init__()

        blocks = []
        for _ in range(depth):
            blocks.append(nn.ModuleList([
                MultiHeadAttention( dim = dim, num_heads=num_heads, head_dim=dim_head, qkv_bias=False, qk_scale=None,
                                    attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_rate),
                FFNBlock_FC(in_dims=dim, internal_dims=int(dim * ffn_expansion), drop_path_rate=drop_path_rate)                    
            ]))
        self.mobilevit_unit = nn.Sequential(*blocks)

    def forward(self, x):
        for attn, ffn in self.mobilevit_unit:
            x = x + attn(x, in_format="BNC", out_format="BNC")
            x = x + ffn(x, in_format="BNC", out_format="BNC")
        return x
    

class MobileViTStage(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, patch_size: Union[int, Tuple[int, int]], attn_channels: int, 
                 out_channels: Union[None, int] = None, attn_depth: int = 2, ffn_expansion: int = 2, drop_path_rate: float = 0.):
        """ Build a stage of MobileViT model.
        Args:
            in_channels: input channel dimensionality.
            kernel_size: Size of the conv kernel for local representations and Fusion parts.
            patch_size: Size of patches. image_size must be divisible by patch_size.
            attn_channels: Number of Transformer input channels.
            out_channels: Number of channels produced by the convolution.
            attn_depth: Number of Transformer blocks.
            ffn_expansion: ratio of ffn hidden channels.
            drop_path_rate: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
        """
        super().__init__()
        self.ph, self.pw = _pair(patch_size)
        out_channels = out_channels or in_channels

        self.local_rep = nn.Sequential(OrderedDict([
            ("conv_kxk", ConvBN(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, activation=nn.SiLU()) ),
            ("conv_1x1", ConvBN(in_channels, attn_channels, kernel_size=1, activation=nn.SiLU()) )
        ]))

        self.transformer = MobileViTUnit(dim=attn_channels, depth=attn_depth, num_heads=4, dim_head=8, 
                                         drop_path_rate=drop_path_rate, ffn_expansion=ffn_expansion)

        self.conv_proj = ConvBN(attn_channels, out_channels, kernel_size=1, activation=nn.SiLU()) 
        self.conv_fusion = ConvBN(in_channels+out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, activation=nn.SiLU()) 
    
    def _interpolate(self, x):
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / self.ph) * self.ph, math.ceil(W / self.pw) * self.pw
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x, (B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.clone()

        # Local representations
        x = self.local_rep(x)
        x, pre_shape = self._interpolate(x)

        # Global representations (Unfold -> Transformer -> Fold)
        _, _, h, w = x.shape
        x = rearrange(x, 'b c (h ph) (w pw) -> b (ph pw) (h w) c', ph=self.ph, pw=self.pw)
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # (B, ph*pw, num_patches, C) -> (B*ph*pw, num_patches, C)
        x = self.transformer(x)
        x = x.reshape(-1, self.ph*self.pw, x.shape[-2], x.shape[-1]) # (B*ph*pw, num_patches, C) -> (B, ph*pw, num_patches, C)
        x = rearrange(x, 'b (ph pw) (h w) c -> b c (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        
        # Fusion
        if x.shape != pre_shape:
            x = F.interpolate(x, size=(pre_shape[2], pre_shape[3]), mode="bilinear", align_corners=False)
        x = self.conv_proj(x)
        x = torch.cat((x, y), 1)
        x = self.conv_fusion(x)
        return x
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [3, 4, 64, 64, 4, 0.1],
        [3, 7, 128, 96, 4, 0.4],
        [5, 8, 32, 128, 2, 0.3],
    ]
    input=torch.randn(3, 96, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        mobileViT = MobileViTStage(input.size(1), kernel_size=opt[0], patch_size=opt[1], attn_channels=opt[2], 
                                   out_channels=opt[3], ffn_expansion=opt[4], drop_path_rate=opt[5])
        mobileViT.to(device)
        mobileViT.eval()
        
        # ori result
        out = mobileViT(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(mobileViT, input)
        summary(copy.deepcopy(mobileViT).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(mobileViT, input, "ori_model.onnx") 

        print('parametrization')
        print(f"Time   : {ori_time:0.2f}s")
        print(f"Flops  : {ori_flops:s}")
        print(f"Params : {ori_params:s}")
        print(f"Memory : {ori_mem:s}")
        print("\n")