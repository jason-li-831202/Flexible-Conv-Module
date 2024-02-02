from torch import nn
from collections import OrderedDict
from torch.nn import init

__all__ = ["SEBlock", "SEBlock_Conv", "ESEBlock_Conv"]
# Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)
# CVPR 2018

# Squeeze_and_Excitation
class SEBlock(nn.Module):
    def __init__(self, channels: int, internal_reduce: int = 16, mode: str = "avg"):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            internal_reduce: the ratio for internal_channels.
            mode: select avg or max pooling.
        """
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1) # GAP
        self.max_pooling = nn.AdaptiveMaxPool2d(1) # GMP
        self.global_pooling = self.max_pooling if mode == "max" else self.avg_pooling

        self.fc_layers = nn.Sequential(OrderedDict([
            ('Squeeze', nn.Linear(in_features = channels, out_features = channels // internal_reduce, bias = False)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('Excitation', nn.Linear(in_features = channels // internal_reduce, out_features = channels, bias = False)),
            ('Sigmoid', nn.Sigmoid()),
        ]))
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
        return "SE_FC"
    
    def forward(self, x):
        b, c, _, _ = x.size() # B, C, H, W
        y = self.global_pooling(x).view(b, c) # (B, C, 1, 1) -> (B, C)
        y = self.fc_layers(y).view(b, c, 1, 1) # (B, C) -> (B, C, 1, 1)
        return x * y.expand_as(x) # (B, C, H, W) * (B, C, H, W)
    
class SEBlock_Conv(nn.Module):
    def __init__(self, channels: int, internal_reduce: int = 16, mode: str = "avg"):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            internal_reduce: the ratio for internal_channels.
            mode: select avg or max pooling.
        """
        super(SEBlock_Conv, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1) # GAP
        self.max_pooling = nn.AdaptiveMaxPool2d(1) # GMP
        self.global_pooling = self.max_pooling if mode == "max" else self.avg_pooling

        self.fc_layers = nn.Sequential(OrderedDict([
            ('Squeeze', nn.Conv2d(channels, channels // internal_reduce, kernel_size=1, bias=False)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('Excitation', nn.Conv2d(channels // internal_reduce, channels, kernel_size=1, bias=False)),
            ('Sigmoid', nn.Sigmoid()),
        ]))

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
        return "SE_Conv"
    
    def forward(self, x):
        avg_x = self.global_pooling(x)
        return x * self.fc_layers(avg_x)
    
class _h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(_h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
class ESEBlock_Conv(nn.Module):
    def __init__(self, channels: int, mode: str = "avg"):
        super(ESEBlock_Conv, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1) # GAP
        self.max_pooling = nn.AdaptiveMaxPool2d(1) # GMP
        self.global_pooling = self.max_pooling if mode == "max" else self.avg_pooling

        self.fc_layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            _h_sigmoid()
        )

    def forward(self, x):
        y = self.global_pooling(x)
        y = self.fc_layers(y)
        return x * y