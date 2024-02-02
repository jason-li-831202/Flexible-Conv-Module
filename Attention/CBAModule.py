import torch
from torch import nn
from torch.nn import init

__all__ = ["CAMBlock_FC", "CAMBlock_Conv", "SAMBlock", "CBAMBlock"]
# CBAM: Convolutional Block Attention Module (https://arxiv.org/abs/1807.06521)
# ECCV 2018

# Channel_Attention_Module
class CAMBlock_FC(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            ratio: the ratio for internal_channels.
        """
        super(CAMBlock_FC, self).__init__()
        self.avg_global_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_global_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        self.sigmoid = nn.Sigmoid()
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

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_x = self.avg_global_pooling(x).view(b, c) # (B, C, 1, 1) -> (B, C) 
        max_x = self.max_global_pooling(x).view(b, c) # (B, C, 1, 1) -> (B, C)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1) # (B, C) -> (B, C, 1, 1)
        return x * v

class CAMBlock_Conv(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            ratio: the ratio for internal_channels.
        """
        super(CAMBlock_Conv, self).__init__()
        self.avg_global_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_global_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
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
                    
    def forward(self, x):
        avg_x = self.avg_global_pooling(x)
        max_x = self.max_global_pooling(x)
        avg_out = self.fc_layers(avg_x)
        max_out = self.fc_layers(max_x)
        v = self.sigmoid(avg_out + max_out)
        return x * v

# Spatial_Attention_Module
class SAMBlock(nn.Module):
    def __init__(self, k: int):
        """ Constructor
        Args:
            k: kernel size.
        """
        super(SAMBlock, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()
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
                    
    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v

# CAM + SAM
class CBAMBlock(nn.Module):
    def __init__(self, spatial_attention_kernel_size: int, 
                 channels: int = None, ratio: int = 16,
                 channel_attention_mode: str = "FC"):
        super(CBAMBlock, self).__init__()
        self.channel_attention_mode = channel_attention_mode
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = CAMBlock_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and ratio != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = CAMBlock_Conv(channels = channels, ratio = ratio)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = SAMBlock(k = spatial_attention_kernel_size)

    @staticmethod
    def get_module_name(self):
        return f"CBM_{self.channel_attention_mode}+SAM"
    
    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x
