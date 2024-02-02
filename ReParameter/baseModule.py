import torch, sys
import numpy as np
from torch import mean, nn
from torch.nn import functional as F

sys.path.append("..")
from Conv.Basic.common import ConvBN

def _ntuple(n):
    def parse(x):
        if not isinstance(x, tuple):
            from itertools import repeat
            return tuple(repeat(x, n))
        return x

    return parse

_pair = _ntuple(2)

class FusedBase(nn.Module):
    def __init__(self, input_channel: int, 
                       output_channel: int, 
                       kernel_size: tuple = None, 
                       groups: int = None, 
                       stride: int = None) :
        super().__init__()
        assert input_channel >= groups, "groups can't bigger than in_channels nums."
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.stride = stride

    # 將branch相Add
    @staticmethod
    def _fuse_addbranch(kernels, biases):
        return sum(kernels), sum(biases)
    
    # 將branch相Concat
    @staticmethod
    def _fuse_concatbranch(kernels, biases):
        return torch.cat(kernels, dim=0), torch.cat(biases)

    def _fuse_conv_1x1_kxk(self, k1, b1, k2, b2):
        if self.groups == 1:
            fused_conv_weight = F.conv2d(k2, k1.permute(1, 0, 2, 3))
            fused_conv_bias = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2
        else :
            k_slices = []
            b_slices = []
            k1_T = k1.permute(1, 0, 2, 3)
            k1_group_width = k1.size(0) // self.groups
            k2_group_width = k2.size(0) // self.groups
            for g in range(self.groups):
                k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) * k1_group_width, :, :]
                k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append(
                    (k2_slice * b1[g * k1_group_width:(g + 1) * k1_group_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
            fused_conv_weight, b_hat = self._fuse_concatbranch(k_slices, b_slices)
            fused_conv_bias = b_hat + b2

        return fused_conv_weight, fused_conv_bias

    def _fuse_fc_bn(self, fc, bn):
        fc_weights = fc
        fc_bias = 0
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps # 设立极小值下限，防止方差过小时进行除法操作而导致溢出

        # 基于方差和极小值计算标准差
        std = (running_var + eps).sqrt()
        # 基于fc和bn的参数值计算出等效融合后的卷积参数值
        fused_fc_weight = (gamma / std).reshape(-1, 1) * fc_weights
        fused_fc_bias = gamma * (fc_bias - running_mean) / std + beta
        return fused_fc_weight, fused_fc_bias
    
    # 將Conv和BN的参数融合到一起
    def _fuse_conv_bn(self, kernel, bn):
        conv_weights = kernel
        conv_bias = 0
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps # 设立极小值下限，防止方差过小时进行除法操作而导致溢出
        
        # 基于方差和极小值计算标准差
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 基于conv和bn的参数值计算出等效融合后的卷积参数值
        if len(t) != conv_weights.size(0):
            # conv 和 bn channel不一致時
            repeat_times = conv_weights.size(0) // len(t)
            fused_conv_weight = conv_weights * t.repeat_interleave(repeat_times, 0)
            fused_conv_bias = (gamma * (conv_bias - running_mean) / std + beta).repeat_interleave(repeat_times, 0)
        else :
            # conv 和 bn channel 一致
            fused_conv_weight =  t * conv_weights
            fused_conv_bias = gamma * (conv_bias - running_mean) / std + beta

        return fused_conv_weight, fused_conv_bias

    def _fuse_conv_bn_module(self, branch: nn.modules):
        if (branch is None):
            return 0, 0
        elif (isinstance(branch, nn.Sequential) or isinstance(branch, ConvBN)):
            conv_weights = branch.conv.weight
            conv_bias = branch.conv.bias if branch.conv.bias !=None else 0 
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps # 设立极小值下限，防止方差过小时进行除法操作而导致溢出
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                self.id_tensor = self._fuse_identity().to(branch.weight.device)

            conv_weights = self.id_tensor
            conv_bias = 0
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        # 基于方差和极小值计算标准差
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # 基于conv和bn的参数值计算出等效融合后的卷积参数值
        if len(t) != conv_weights.size(0):
            # conv 和 bn channel不一致時
            repeat_times = conv_weights.size(0) // len(t)
            fused_conv_weight = conv_weights * t.repeat_interleave(repeat_times, 0)
            fused_conv_bias = (gamma * (conv_bias - running_mean) / std + beta).repeat_interleave(repeat_times, 0)
        else :
            # conv 和 bn channel 一致
            fused_conv_weight =  t * conv_weights
            fused_conv_bias = gamma * (conv_bias - running_mean) / std + beta
        
        return fused_conv_weight, fused_conv_bias
    
    # 將1x1的Cconv變成kxk的Conv參數
    def _pad_kernel(self, kernel: nn.Conv2d, kernel_size: tuple = None):
        if kernel_size is None :
            kernel_size = self.kernel_size
        
        if (kernel is None):
            return 0
        else:
            if (kernel.size(2) == kernel.size(3) and kernel_size[0] == kernel_size[1]) :
                pad = (kernel_size[0] - kernel.size(2)) // 2
                return F.pad(kernel, [pad] * 4)
            w_pad = (kernel_size[0] - kernel.size(2)) // 2
            h_pad = (kernel_size[1] - kernel.size(3)) // 2
            
            return F.pad(kernel, [h_pad, h_pad, w_pad, w_pad])

    # 將avgpooling變成kxk的Conv參數
    def _fuse_avgpooling(self, avg: nn.AvgPool2d):
        input_dim = self.output_channel // self.groups
        kernel_value = np.zeros((self.output_channel, input_dim, *avg.kernel_size), dtype=np.float32) 
        for i in range(self.output_channel):
            kernel_value[i, i % input_dim, :, :]= 1.0 / torch.prod(torch.tensor(avg.kernel_size)) 

        return  torch.from_numpy(kernel_value) , torch.tensor([0])
    
    # 將identity變成kxk的Conv參數
    def _fuse_identity(self, kernel_size: tuple = None):
        if kernel_size is None :
            kernel_size = self.kernel_size
        
        input_dim = self.input_channel // self.groups
        kernel_value = np.zeros((self.input_channel, input_dim, *kernel_size), dtype=np.float32)
        for i in range(self.input_channel): #[i, i, center_pos, center_pos]
            kernel_value[i, i % input_dim, int(np.floor(kernel_size[0]/2)), int(np.floor(kernel_size[1]/2))] = 1.0
        
        return torch.from_numpy(kernel_value)
