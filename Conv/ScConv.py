import torch
import torch.nn.functional as F
import torch.nn as nn 

__all__ = ["SCReconstructConv2d"]
# SCConv: Spatial and Channel Reconstruction Convolution for Feature Redundancy
# CVPR 2023

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps  = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view( N, self.group_num, -1)
        mean = x.mean( dim = 2, keepdim = True)
        std = x.std ( dim = 2, keepdim = True)
        x = (x - mean) / (std+self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

# Spatial Reconstruction Unit   
class _SRUnit(nn.Module):
    def __init__(self, channels: int, group_num: int = 16, gate_threshold: float = 0.5, torch_gn: bool = True):
        super().__init__()
        
        self.gn = nn.GroupNorm( num_channels = channels, num_groups = group_num ) \
                  if torch_gn else GroupBatchnorm2d(c_num = channels, group_num = group_num)
        self.gate_threshold = gate_threshold
        self.sigomid = nn.Sigmoid()

    def forward(self,x):
        gn_x         = self.gn(x)
        w_gamma      = self.gn.weight/sum(self.gn.weight)
        w_gamma      = w_gamma.view(1, -1, 1, 1)
        reweigts     = self.sigomid( gn_x * w_gamma )
        # Gate
        info_mask    = torch.where(reweigts > self.gate_threshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        noninfo_mask = torch.where(reweigts > self.gate_threshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        return self.reconstruct(info_mask*x, noninfo_mask*x)
    
    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, torch.div(x_1.size(1), 2, rounding_mode='trunc'), dim=1)
        x_21, x_22 = torch.split(x_2, torch.div(x_2.size(1), 2, rounding_mode='trunc'), dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)

# Channel Reconstruction Unit
class _CRUnit(nn.Module):
    '''
    alpha: 0 < alpha < 1
    '''
    def __init__(self, channels: int, alpha: float = 1/2, squeeze_radio: int = 2 , groups: int = 2, group_kernel_size: int = 3):
        super().__init__()
        self.up_channel     = up_channel   = int(alpha*channels)
        self.low_channel    = low_channel  = channels-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)

        # Up Transform
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, channels,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = groups)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, channels,kernel_size=1, bias=False)
        
        # Down Transform
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, channels-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up, low = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up, low = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1, out2 = torch.split(out, torch.div(out.size(1), 2, rounding_mode='trunc'), dim=1)
        return out1+out2


class SCReconstructConv2d(nn.Module):
    """
    Args:
        channels: Number of channels in the input image
        spatial_group_num: Number of groups for Spatial Reconstruct Unit. Default: 4
        gate_threshold: gate threshold for Spatial and Reconstruct Unit. Default: 0.5
        alpha: the ratio of up/low channels. Default: 0.5
        channel_ratio: the ratio of internal channels for Channel Reconstruction Unit. Default: 0.5
        channel_group_num: Number of groups for Channel Reconstruction Unit. Default: 2
        channel_kernel_size: Size of the conv kernel for Channel Reconstruction Unit.
    """
    def __init__(self,  channels: int,
                        spatial_group_num: int = 4,
                        gate_threshold: float = 0.5,
                        alpha: float = 0.5,
                        channel_ratio: int = 2 ,
                        channel_group_num: int = 2,
                        channel_kernel_size: int = 3):
                 
        super().__init__()
        self.SR_part = _SRUnit( channels = channels, 
                                group_num = spatial_group_num,  
                                gate_threshold = gate_threshold )
        self.CR_part = _CRUnit( channels = channels, 
                                alpha = alpha, 
                                squeeze_radio = channel_ratio,
                                groups = channel_group_num ,
                                group_kernel_size = channel_kernel_size )
    
    def forward(self,x):
        x = self.SR_part(x)
        x = self.CR_part(x)
        return x
