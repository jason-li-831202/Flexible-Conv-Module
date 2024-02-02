import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

# SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS (https://arxiv.org/abs/2102.00240)
# ICASSP 2021
__all__ = ["ShuffleAttBlock"]

class ShuffleAttBlock(nn.Module):

    def __init__(self, channels: int = 512, groups: int = 8):
        super().__init__()
        self.groups = groups

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_weight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.channel_bias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))

        self.gn = nn.GroupNorm(channels // (2 * groups), channels // (2 * groups))
        self.spatial_weight = Parameter(torch.zeros(1, channels // (2 * groups), 1, 1))
        self.spatial_bias = Parameter(torch.ones(1, channels // (2 * groups), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def get_module_name():
        return "ShuffleAtt"
    
    @staticmethod
    def channel_shuffle(x, c_groups):
        bs, _, f_h, f_w = x.shape
        x = x.reshape(bs, c_groups, -1, f_h, f_w)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(bs, -1, f_h, f_w) # flatten

    def forward(self, x):
        origin_shape = x.size() # bs, C, H, W
        bs, _, f_h, f_w = origin_shape

        # group into subfeatures
        x = x.view(bs*self.groups, -1, f_h, f_w) # bs*G, c//G, H, W

        # channel_split
        channel_part, spatial_part = x.chunk(2, dim=1) # (bs*G, c//G, H, W) -> [(bs*G, c//(2*G), H, W), (bs*G, c//(2*G), H, W)]

        # channel attention
        x_channel = self.global_avg_pooling(channel_part) # bs*G, c//(2*G), 1, 1
        x_channel = self.channel_weight*x_channel + self.channel_bias # bs*G, c//(2*G), 1, 1
        x_channel = channel_part*self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(spatial_part) # bs*G, c//(2*G), 1, 1
        x_spatial = self.spatial_weight*x_spatial + self.spatial_bias # bs*G, c//(2*G), H, W
        x_spatial = spatial_part*self.sigmoid(x_spatial) # bs*G, c//(2*G), H, W

        # concatenate along channel axis
        out = torch.cat([x_channel,x_spatial], dim=1) # bs*G, c//G,, H, W
        out = out.contiguous().view(bs, -1, f_h, f_w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out