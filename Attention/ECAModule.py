import torch, math
from torch import nn
from torch.nn import init

# ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (https://arxiv.org/abs/1910.03151)
# CVPR 2020
__all__ = ["ECABlock"]

# Efficient_Channel_Attention_Module
class ECABlock(nn.Module):  
    def __init__(self, channels: int, internal_reduce: int = 2, b: int = 1):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            internal_reduce: the ratio for internal_channels.
        """
        super(ECABlock, self).__init__()

        self.kernel_size = int(abs((math.log(channels, 2) + b) / internal_reduce))  # 计算1d卷积核尺寸
        self.kernel_size = self.kernel_size if self.kernel_size % 2 else self.kernel_size + 1  

        self.avg_global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, self.kernel_size, 1, padding = int(self.kernel_size / 2), bias=False) 
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
                    
    @staticmethod
    def get_module_name():
        return "ECA"
    
    def forward(self, x):
        weights = self.avg_global_pooling(x) # (B, C, W, H) -> (B, C, 1, 1)
        weights = self.conv(weights.squeeze(-1).transpose(-1, -2)) # (B, C, 1, 1) -> (B, C, 1) -> (B, 1, c) -> (B, 1, c)
        weights = self.sigmoid(weights.transpose(-1, -2).unsqueeze(-1))  # (B, 1, c) -> (B, C, 1) -> (B, C, 1, 1) -> (B, C, 1, 1)
        return weights * x  # 将计算得到的weights与输入的feature map相乘
