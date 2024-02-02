import torch, math
from torch import nn
from torch.nn import init

try :
    from Conv.Basic.common import ConvBN
except :
    import sys
    sys.path.append("..")
    from Conv.Basic.common import ConvBN

# Selective Kernel Networks (https://arxiv.org/abs/1903.06586)
# CVPR 2019
__all__ = ["SKBlock"]

# Selective_Kernel_Module
class SKBlock(nn.Module):
    def __init__(self, channels: int, multi_receptive_field: int = 2, 
                 groups: int = 32, stride: int = 1, 
                 internal_reduce: int = 16, min_internal_dims: int = 32):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            multi_receptive_field: the number of branchs.
            groups: num of convolution groups.
            stride: stride, default 1.
            internal_reduce: the ratio for compute d, the length of z.
            min_internal_dims: the minimum dim of the vector z in paper, default 32.
        """
        super(SKBlock, self).__init__()
        internal_channels = max( channels // internal_reduce, min_internal_dims)
        self.M = multi_receptive_field
        self.channels = channels
        
        # Split Part
        self.multi_branch_convs = nn.ModuleList([])
        for i in range(self.M):
            self.multi_branch_convs.append(nn.Sequential(
                ConvBN(channels, channels, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=groups, bias=False),
                nn.ReLU(inplace=True)
            ))
        
        # Fuse Part
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(ConvBN(channels, internal_channels, kernel_size=1, stride=1, bias=False),
                                nn.ReLU(inplace=True))
        
        # Select Part
        self.multi_branch_fcs = nn.ModuleList([])
        for i in range(self.M):
            self.multi_branch_fcs.append(nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
                    
    @staticmethod
    def get_module_name():
        return "SK"
    
    def forward(self, inputs):
        origin_shape = inputs.size() # B, C, H, W
        batch_size = origin_shape[0]

        # 各分支的卷積計算
        feats = [conv(inputs) for conv in self.multi_branch_convs] # [B, C, f_w, f_h]*m 
        feats = torch.cat(feats, dim=1) # B, m*C, f_w, f_h
        feats = feats.view(batch_size, self.M, self.channels, feats.shape[2], feats.shape[3]) # B, m, C, f_w, f_h

        feats_U = torch.sum(feats, dim=1) # B, C, f_w, f_h
        feats_S = self.global_avg_pooling(feats_U) # B, C, 1, 1
        feats_Z = self.fc(feats_S) # B, C, 1, 1

        # 各分支的全連結層
        attention_vectors = [fc(feats_Z) for fc in self.multi_branch_fcs] # [B, C, 1, 1]*m
        attention_vectors = torch.cat(attention_vectors, dim=1) # B, m*C, 1, 1
        attention_vectors = attention_vectors.view(batch_size, self.M, self.channels, 1, 1) # B, m, C, f_w, f_h
        attention_vectors = self.softmax(attention_vectors)

        return torch.sum(feats*attention_vectors, dim=1) # B, C, f_w, f_h