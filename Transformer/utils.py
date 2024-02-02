import torch
from torch import nn
from torch.nn import functional as F

def view_format(x, in_type, out_type, shape=None):
    if (in_type == "BCHW"):
        B, C, H, W = x.shape
        if (out_type == "BNC"):
            # flatten: (B, C, H, W) -> (B, C, H*W) 
            # transpose: (B, C, H*W) -> (B, H*W, C)
            x = x.flatten(2).transpose(1, 2)
            return x, (B, C, H, W)
        elif (out_type == "BHWC"):
            # (B, C, H, W) -> (B, H, W, C)
            x = x.permute(0, 2, 3, 1)
            return x
        else :
            return x
    elif (in_type == "BHWC"):
        B, H, W, C = x.shape
        if (out_type == "BNC"):
            # (B, H, W, C) -> (B, N, C)
            return x.view(B, H*W, C), (B, H, W, C)
        elif (out_type == "BCHW"):
            # (B, H, W, C) -> (B, C, H, W)
            return x.permute(0, 3, 1, 2)
        else :
            return x
    else:
        if (out_type == "BHWC"):
            assert shape != None, "ref shape can't be empty."
            B, N, C = x.shape
            _, H, W, _ = shape
            # (B, N, C) -> (B, H, W, C)
            return x.view(B, H, W, C)
        elif (out_type == "BCHW"):
            assert shape != None, "ref shape can't be empty."
            B, N, C = x.shape
            _, _, H, W = shape
            # (B, N, C) -> (B, C, N) -> (B, C, H, W)
            return x.transpose(1, 2).reshape(B, C, H, W)
        else :
            return x
            
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self._drop_path(x, self.drop_prob, self.training)
    
    @staticmethod
    def _drop_path(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        output = x.div(keep_prob) * random_tensor
        return output
    

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x