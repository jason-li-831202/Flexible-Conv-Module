import torch, sys
from torch import nn
from torch.nn import init

sys.path.append("..")
from Conv.Basic.PyConv import PyConv2d
from Attention.SEModule import SEBlock_Conv

__all__ = ["PSABlock"]

class SEBlock_nonFusion(SEBlock_Conv):
    def __init__(self, channels: int, internal_reduce: int = 16, mode: str = "avg"):
        super(SEBlock_nonFusion, self).__init__(channels, internal_reduce, mode)

    def forward(self, x):
        avg_x = self.global_pooling(x)
        return self.fc_layers(avg_x)
    
# Pyramid Squeeze Attention
class PSABlock(PyConv2d):
    def __init__(self, in_channels, out_channels, multi_kernels=[3, 5, 7, 9], multi_groups=[1, 4, 8, 16], stride=1):
        super(PSABlock, self).__init__(in_channels, out_channels, multi_kernels, multi_groups, stride, 1, False)
        if len(multi_kernels) != 4:
            raise ValueError("Num of kernels need to be 4!")
        
        self.se = SEBlock_nonFusion(out_channels // 4)
        self.split_channel = out_channels // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, C, H, W = x.shape

        feats = []
        attns = []
        for level in self.pyConv:
            feat = level(x)
            feats.append(feat)
            attn = self.se(feat)
            attns.append(attn)

        feats = torch.cat(feats, dim=1)
        B, C, f_h, f_w = feats.shape

        feats_vectors = feats.view(B, 4, self.split_channel, f_h, f_w)
        attns = torch.cat(attns, dim=1)
        attention_vectors = attns.view(B, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        weights = feats_vectors * attention_vectors
        return weights.reshape(B, 4*self.split_channel, f_h, f_w)
