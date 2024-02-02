import torch, sys
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple
from collections import OrderedDict

sys.path.append("..")
from Conv.Basic.common import ConvBN
from Transformer.FFNModule import FFNBlock_Conv
from Transformer.utils import DropPath, LayerNorm

__all__ = ["WaveUnit", "WaveStage"]
# An Image Patch is a Wave: Quantum Inspired Vision MLP ( https://arxiv.org/abs/2111.12294 )
# CVPR 2022

class _PhaseAwareTokenMixing(nn.Module):
    def __init__(self, dim, qkv_bias=False, drop_rate=0., mode='fc'):
        super().__init__()

        self.mode = mode
        if mode=='fc':
            self.theta_h_conv = ConvBN(dim, dim, kernel_size=1, stride=1, bias=True, activation=nn.ReLU())
            self.theta_w_conv = ConvBN(dim, dim, kernel_size=1, stride=1, bias=True, activation=nn.ReLU())
        else:
            self.theta_h_conv = ConvBN(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, activation=nn.ReLU())
            self.theta_w_conv = ConvBN(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, activation=nn.ReLU())

        self.fc_h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias) 
        self.fc_c = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, kernel_size=(1, 7), stride=1, padding=(0, 7//2), groups=dim, bias=False) 
        self.tfc_w = nn.Conv2d(2*dim, dim, kernel_size=(7, 1), stride=1, padding=(7//2, 0), groups=dim, bias=False)  
        self.reweight = FFNBlock_Conv(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.dropout = nn.Dropout(drop_rate)  

    def forward(self, x):
        B, C, H, W = x.shape

        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)
        x_h = self.fc_h(x)
        x_w = self.fc_w(x)      
        x_h = torch.cat([x_h*torch.cos(theta_h), x_h*torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w*torch.cos(theta_w), x_w*torch.sin(theta_w)], dim=1)


        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)         
        return self.dropout(x)
    

class WaveUnit(nn.Module):
    def __init__(self, dims, attn_bias=False, attn_drop_rate=0., mode='fc', ffn_expansion = 4., 
                  drop_path_rate: float = 0.):
        """ Constructor
        Args:
            dims: Number of input channels.
            attn_bias: Whether to use bias for Phase-Aware Token-Mixing .
            attn_drop_rate: Whether to use dropout rate for Phase-Aware Token-Mixing .
            ffn_expansion: ratio of ffn hidden channels.
            drop_path_rate: Stochastic depth rate. Default: 0.0
        """
        super().__init__()

        self.attn_block = nn.Sequential( nn.BatchNorm2d(dims),
                                         _PhaseAwareTokenMixing(dims, qkv_bias=attn_bias, drop_rate=attn_drop_rate, mode=mode))

        self.ffn_block = FFNBlock_Conv(in_channels=dims, internal_channels=int(dims * ffn_expansion) )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn_block(x)) 
        x = x + self.drop_path(self.ffn_block(x))
        return x
    

class WaveStage(nn.Module):
    def __init__(self, channels: int, num_units: int, ffn_expansion: int = 3., attn_bias: bool = False, attn_drop_rate: float = 0.,
                 drop_path_rates: Union[float, Tuple[float, float]] = 0., mode: str = 'fc'):
        """ Build a stage of WaveStage model.
        Args:
            channels: input channel dimensionality.
            num_units: Number of unit modeules in this stage.
            ffn_expansion: ratio of ffn hidden channels.
            attn_bias: Whether to use bias for Phase-Aware Token-Mixing .
            attn_drop_rate: Whether to use dropout rate for Phase-Aware Token-Mixing .
            drop_path_rates: We set block-wise drop-path rate. The higher level blocks are more likely to be dropped.
        """
        super().__init__()
        if isinstance(drop_path_rates, list) :
            assert num_units==len(drop_path_rates), "num of drop path list must be consistent with num_units"
        
        units = []
        for i in range(num_units):
            block_drop_path = drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates
            units.append(WaveUnit(dims=channels, ffn_expansion=ffn_expansion, attn_bias=attn_bias,
                                attn_drop_rate=attn_drop_rate, drop_path_rate=block_drop_path, mode=mode))
        self.wave_stage = nn.ModuleList(units)
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
        for unit in self.wave_stage:
            x = unit(x)
        return x
    

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options =[
        [3, 4, 0.0],
        [3, 3, 0.5],
        [6, 3, 0.2],
    ]
    input=torch.randn(3, 96, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        convNeXt = WaveStage(input.size(1), num_units=opt[0], ffn_expansion=opt[1], drop_path_rates=opt[2])
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
