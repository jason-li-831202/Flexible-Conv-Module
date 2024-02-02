import torch, math
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

try :
    from Conv.Basic.common import ConvBN
except :
    import sys
    sys.path.append("..")
    from Conv.Basic.common import ConvBN

# Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks (https://arxiv.org/abs/2105.02358)
# CVPR 2021
__all__ = ["EABlock"]

# External Attention Module
class EABlock(nn.Module):
    ''' Constructor
    Args:
        channels: The input and output channel number.
        internal_dims: the dim of the fc dims, default 64.
    '''
    def __init__(self, channels: int, internal_dims: int = 64):
        super(EABlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 1)

        self.mk = nn.Conv1d(channels, internal_dims, 1, bias=False)

        self.mv = nn.Conv1d(internal_dims, channels, 1, bias=False)
        self.mv.weight.data = self.mk.weight.data.permute(1, 0, 2)        
        
        self.conv2 = ConvBN(channels, channels, kernel_size=1, bias=False)      
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def get_module_name():
        return "EA"
    
    def forward(self, x):
        if len(x.size()) > 3:
            bs, c, f_h, f_w = x.size()
            identity = x
            x = self.conv1(x)
            x = x.view(bs, c, f_h*f_w) # bs, C, H*W

        attn = self.mk(x) # b, internal_d, H*W
        attn = F.softmax(attn, dim=-1) # b, internal_d, H*W
        
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True)) #  # b, internal_d, H*W
        x = self.mv(attn) # b, c, H*W

        if len(x.size()) > 3:
            x = x.view(bs, c, f_h, f_w)
            x = self.conv2(x) # bs, C, H, W
            x = x + identity
            x = F.relu(x)

        return x
