import torch, sys
from collections import OrderedDict
from torch.nn import functional as F
from torch import nn

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN
from Transformer.FFNModule import FFNBlock_Conv

__all__ = ["RepMLPBlock", "RepMLPUnit"]
# RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (https://arxiv.org/abs/2112.11081)
# CVPR 2022

class _Reshape(nn.Module):
    def __init__(self, *args):
        super(_Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class RepMLPBlock(FusedBase):
    """
             |
             |--------------------------------
             |                               |
        ------------                         |
        |  Reshape |                         |
        ------------                         |
             |                               |
             |----------------               |
             |               |               |
             |               |               | 
        ------------    -----------    ------------
        |  Channel |    |  Local  |    |  Global  | 
        ------------    -----------    ------------
             |               |               |  
            Add --------------               |
             |                               |
            Mul ------------------------------
             |

                                    |
                                    V

             |
             |--------------------------------
             |                               |
        ------------                         |
        |  Reshape |                         |
        ------------                         |
             |                               |
             |                               | 
        ------------                   ------------
        |  Channel |                   |  Global  | 
        ------------                   ------------
             |                               |  
             |                               |
             |                               |
            Mul ------------------------------
             |
    """

    def __init__(self, channels: int, patch_h: int, patch_w: int,
                 global_perceptron_reduce: int = 1, num_sharesets: int = 1,
                 local_conv_kernels: list = None, fused: bool = False):
        """ Constructor
        Args:
            channels: input channel dimensionality.
            patch_h: patch height pixels.
            patch_w: patch width pixels.
            global_perceptron_reduce: adjust the internal dims between fc1 and fc2 according to the channels.
            num_sharesets: the number of groups used by conv in the local perceptron.
            local_conv_kernels: the size of kernel list used by conv in the local perceptron.

        """
        super(RepMLPBlock, self).__init__(channels, channels, groups=num_sharesets)
        self.h = patch_h
        self.w = patch_w
        self.num_sharesets = num_sharesets

        self.local_conv_kernels = local_conv_kernels
        self.fused = fused
    
        # assert in_channels == out_channels
        # --------------- Global Perceptron ------------------
        fc_internal_dims = int(self.input_channel // global_perceptron_reduce)
        self.__global_perceptron_dims = self.input_channel
        self.fc1_fc2 = nn.Sequential(OrderedDict([
            ('avg', nn.AdaptiveAvgPool2d(1)),
            ('fc1', nn.Conv2d(self.__global_perceptron_dims, fc_internal_dims, kernel_size=1, stride=1, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2' ,nn.Conv2d(fc_internal_dims, self.__global_perceptron_dims, kernel_size=1, stride=1, bias=True)),
            ('sigmoid', nn.Sigmoid()),
        ]))

        # --------------- Channel Perceptron ---------------
        self.__channel_perceptron_dims = self.num_sharesets * self.h * self.w
        self.fc3_bn = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(self.__channel_perceptron_dims, self.__channel_perceptron_dims, 
                               kernel_size = 1, stride = 1, padding = 0, groups = self.num_sharesets, bias = self.fused)),
            ('Reshape', nn.Identity() if self.fused else _Reshape(self.num_sharesets, self.h, self.w)),              
            ('bn', nn.Identity() if self.fused else nn.BatchNorm2d(self.num_sharesets))
        ]))

        # --------------- Local Perceptron ---------------
        if not self.fused and self.local_conv_kernels is not None:
            for k in self.local_conv_kernels:
                local_conv = ConvBN(self.num_sharesets, self.num_sharesets, kernel_size=k, padding=k//2, groups = self.num_sharesets, bias=False)
                self.__setattr__('local_conv{}x{}'.format(k, k), local_conv)

    def _conv_to_fc(self,conv_kernel, conv_bias):
        I = torch.eye(self.h * self.w).repeat(1, self.num_sharesets).reshape(self.h * self.w, self.num_sharesets, self.h, self.w).to(conv_kernel.device)
        fc_k = F.conv2d(I, conv_kernel, padding=(conv_kernel.size(2)//2,conv_kernel.size(3)//2), groups=self.num_sharesets)
        fc_k = fc_k.reshape(self.h * self.w, self.num_sharesets * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias

    def forward(self, inputs) :
        origin_shape = inputs.size() # B, C, H, W
        h_parts = torch.div(origin_shape[2], self.h, rounding_mode='trunc')
        w_parts = torch.div(origin_shape[3], self.w, rounding_mode='trunc')

        partitions = inputs.reshape(-1, self.input_channel, h_parts, self.h, w_parts, self.w) # B, C, h_part, h, w_part, w
        partitions = partitions.permute(0, 2, 4, 1, 3, 5) # B, h_part, w_part, C, h, w

        # Feed inputs into Global Perceptron
        global_out_perceptron = self.fc1_fc2(inputs) 
        global_out_perceptron = global_out_perceptron.view(-1, self.__global_perceptron_dims, 1, 1) # B, C, 1, 1

        # Feed partition map into Channel Perceptron
        channel_in_perceptron = partitions.reshape(-1, self.__channel_perceptron_dims, 1, 1) # B*h_part*w_part, num_sharesets*h*w, 1, 1
        channel_out_perceptron = self.fc3_bn(channel_in_perceptron) # B*h_part*w_part, num_sharesets*h*w, 1, 1
        partitions_out = channel_out_perceptron.reshape(-1, h_parts, w_parts, self.num_sharesets, self.h, self.w) # B, h_part, w_part, num_sharesets, h, w

        # Feed partition map into Local Perceptron
        if(self.local_conv_kernels is not None and not self.fused):
            local_in_perceptron = partitions.reshape(-1, self.num_sharesets, self.h, self.w) #B*h_part*w_part, num_sharesets, h, w
            local_out_perceptron = 0
            for k in self.local_conv_kernels:
                local_conv = self.__getattr__('local_conv{}x{}'.format(k, k))
                local_out_perceptron += local_conv(local_in_perceptron) # B*h_part*w_part, O, h, w
            local_out_perceptron = local_out_perceptron.view(-1, h_parts, w_parts, self.num_sharesets, self.h, self.w) #B, h_part, w_part, num_sharesets, h, w
            partitions_out += local_out_perceptron


        partitions_out = partitions_out.permute(0, 3, 1, 4, 2, 5) # B, num_sharesets, h_part, h, w_part, w
        return partitions_out.reshape(*origin_shape) * global_out_perceptron # (B, O, h_part*h, w_part*w) * (B, O, 1, 1)

    def merge_kernel(self):
        if (self.fused): return 0
        self.fused = True
        partition_weight, partition_bias = self._get_equivalent_kernel_bias()
        
        # --------------- Global Perceptron ---------------
        # None
        # --------------- Channel Perceptron ---------------
        self.__delattr__('fc3_bn')
        self.fc3_bn = nn.Conv2d(self.__channel_perceptron_dims, self.__channel_perceptron_dims, 
                               kernel_size = 1, stride = 1, padding = 0, groups = self.num_sharesets, bias = True)
        self.fc3_bn.weight.data = partition_weight
        self.fc3_bn.bias.data = partition_bias

        # --------------- Local Perceptron ---------------
        if(self.local_conv_kernels is not None):
            for k in self.local_conv_kernels:
                self.__delattr__('local_conv{}x{}'.format(k, k))

    def _get_equivalent_kernel_bias(self):
        # --------------- Channel Perceptron ---------------
        partitions_weight, partitions_bias = self._fuse_conv_bn_module(self.fc3_bn)

        # --------------- Local Perceptron ---------------
        if(self.local_conv_kernels is not None):
            max_kernel = max(self.local_conv_kernels)
            total_kernel, total_bias = 0, 0
            for k in self.local_conv_kernels:
                tmp_branch = self.__getattr__('local_conv{}x{}'.format(k, k))
                tmp_weight, tmp_bias = self._fuse_conv_bn_module(tmp_branch)
                if (k != max_kernel) :
                    tmp_weight = self._pad_kernel(tmp_weight, (max_kernel, max_kernel))
                total_kernel += tmp_weight
                total_bias += tmp_bias
            local_weight_perceptron, local_bias_perceptron = self._conv_to_fc(total_kernel, total_bias)
            partitions_weight += local_weight_perceptron.reshape_as(partitions_weight)
            partitions_bias += local_bias_perceptron

        return partitions_weight, partitions_bias

class RepMLPUnit(nn.Module):

    def __init__(self, channels: int, patch_h: int, patch_w: int, 
                 global_perceptron_reduce, num_sharesets=1, 
                 local_conv_kernels=None, ffn_ratio=4, fused=False):
        super().__init__()
        self.repmlp_pre_bn = nn.BatchNorm2d(channels)
        self.repmlp_block = RepMLPBlock(channels=channels, patch_h=patch_h, patch_w=patch_w,
                                        global_perceptron_reduce=global_perceptron_reduce, num_sharesets=num_sharesets, 
                                        local_conv_kernels=local_conv_kernels, fused=fused)
        self.ffn_block = FFNBlock_Conv(channels, channels * ffn_ratio)
        
    def forward(self, x):
        y = x + self.repmlp_block(self.repmlp_pre_bn(x)) # TODO: use droppath?
        z = y + self.ffn_block(y)
        return z

    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, RepMLPBlock):
                module.merge_kernel()

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = [
        [16, 2, 8, [1, 3, 5, 7]],
        [16, 2, 4, [1]],
        [8, 4, 4, [1, 3]],
    ]
    input=torch.randn(16, 32, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        patch_h = patch_w = opt[0]        # patch height/width ex : 64, 32, 16, 8
        global_perceptron_reduce = opt[1] # reduction ratio
        num_sharesets = opt[2]            # groups ex: 4, 8, 16, 32
        local_conv_kernels = opt[3]       # kernel list
        rep_mlp = RepMLPUnit(input.size(1), patch_h, patch_w, global_perceptron_reduce, 
                                    num_sharesets, local_conv_kernels=local_conv_kernels)
        rep_mlp.to(device)
        rep_mlp.eval()

        # ori result
        out = rep_mlp(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(rep_mlp, input)
        save_model_to_onnx(rep_mlp, input, "ori_model.onnx") 

        # fuse result
        rep_mlp.merge_kernel()
        out2 = rep_mlp(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(rep_mlp, input)
        save_model_to_onnx(rep_mlp, input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")
