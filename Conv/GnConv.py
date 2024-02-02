import torch
from torch import nn
import torch.nn.functional as F

try :
    from Transformer.utils import LayerNorm
except :
    import sys
    sys.path.append("..")
    from Transformer.utils import LayerNorm


__all__ = ["RecursiveGatedConv2d"]

class _GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        nn.init.trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class RecursiveGatedConv2d(nn.Module):
    """
    Args:
        channels: Number of channels in the input image
        order: Number of order spatial
        scale: alpha
        use_globalfilter: Whether to use GlobalLocalFilter. Default: False
        h: h for GlobalLocalFilter
        w: w for GlobalLocalFilter
    """
    def __init__(self,  channels: int, 
                        order: int = 3, 
                        scale: float = 1.0, 
                        use_globalfilter: bool = False, 
                        h: int = 14, 
                        w: int = 8):
        super().__init__()
        self.order = order # 空间交互的阶数
        self.scale = scale # 缩放系数，对应公式3.3中的$\alpha$
        self.dims = [channels // 2 ** i for i in range(order)] # 将2C在不同阶的空间上进行切分，对应公式3.2
        self.dims.reverse()

        self.proj_in_linear = nn.Conv2d(channels, 2*channels, kernel_size=1)

        if not use_globalfilter:
            self.dw_conv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7-1)//2, groups=sum(self.dims))
        else:
            self.dw_conv = _GlobalLocalFilter(sum(self.dims), h = h, w = w) # 在全特征上进行卷积，多在后期使用


        self.pw_convs = nn.ModuleList([
            nn.Identity(),
            *[nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=1) for i in range(order - 1)] # 高阶空间交互过程中使用的卷积模块，对应公式3.4
        ])
        self.proj_out_linear = nn.Conv2d(channels, channels, kernel_size=1)

        print('[RecursiveGatedConv2d]', order, 'order with dims =', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in_linear(x) # B, 2C, H, W
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1) # (B, dims[0], H, W), (B, sum(dims), H, W)

        dw_abc = self.dw_conv(abc) * self.scale # (B, sum(dims), H, W)
        dw_list = torch.split(dw_abc, self.dims, dim=1) # ((B, dims[i], H, W) for i in order)

        for i in range(self.order):
            pwa = self.pw_convs[i](pwa) * dw_list[i]
        return self.proj_out_linear(pwa) # (B, C, H, W)


