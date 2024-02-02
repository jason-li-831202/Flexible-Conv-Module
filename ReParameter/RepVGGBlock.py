import torch, sys
from torch import nn
from typing import Union, Tuple

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN
from Attention import SEModule, coordAttModule, simAModule

__all__ = ["RepVGGBlock", "RepVGGStage"]
# RepVGG: Making VGG-style ConvNets Great Again (https://arxiv.org/abs/2101.03697)
# CVPR 2021

class RepVGGBlock(FusedBase):
    """
            |
            |----------------------------------
            |   (1)          |  (2)           | (3)
        ------------    ------------          |    
        | kxk Conv |    | 1x1 Conv |          |      
        ------------    ------------    ------------ 
        |    BN    |    |   BN     |    |    BN    | 
        ------------    ------------    ------------ 
            |                |               |       
            |---------------------------------
            |

                            |
                            V

            |
            |---------------------------------
            |   (1)          |  (2)          | (3)
        ------------    ------------   ------------
        | kxk Conv |    | kxk Conv |   | kxk Conv |
        ------------    ------------   -----------
            |                |               |
            |---------------------------------
            |

                            |
                            V

                            |
                            |
                            |
                       ------------
                       | kxk Conv |
                       ------------
                            |
                            |
                            |                      

    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), 
                 stride=1, dilation=1, groups=1, 
                 fused=False, use_nonlinear=True, use_bias=False, use_se=False):
        """ Initialization of the class.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 1
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input channels to output channels. Default: 1
            fused: Whether to be deploy status or training status. Default: False
            use_nonlinear: Whether to use activate function.
            use_se: Whether to use se. Default: False
            use_bias: Whether to use conv bias.
        """
        super(RepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, groups, stride)
        self.padding =  (dilation*(self.kernel_size[0]-1)//2, dilation*(self.kernel_size[1]-1)//2)

        self.dilation = dilation
        self.fused = fused
        self.use_bias = use_bias
        self.activation =  nn.ReLU() if use_nonlinear else nn.Identity()
        self.se = SEModule.SEBlock_Conv(out_channels, out_channels // 16) if use_se else nn.Identity()

        if fused:
            self.rbr_reparam = self._fuse_branch()
        else:
            self.rbr_kxk = ConvBN(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, dilation=dilation, padding=self.padding, groups=groups, bias=use_bias)
            self.rbr_1x1 = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=use_bias) \
                            if self.kernel_size != (1, 1) else None
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                                if out_channels == in_channels and stride == 1 else None

            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):

        if (self.fused):
            return self.activation(self.se(self.rbr_reparam(inputs)))

        rbr_kxk_out = self.rbr_kxk(inputs)
        rbr_1x1_out = 0 if (self.rbr_1x1 == None) else self.rbr_1x1(inputs)
        rbr_identity_out = 0 if (self.rbr_identity == None) else self.rbr_identity(inputs)

        return self.activation(self.se( rbr_kxk_out + 
                                        rbr_1x1_out + 
                                        rbr_identity_out))

    def merge_kernel(self):
        if (self.fused): return 0
        self.fused = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = self._fuse_branch()
        
        self.rbr_reparam.weight.data = kernel
        if (self.use_bias) :
            self.rbr_reparam.bias.data = bias
        self.del_ori_bracnch()

    def _fuse_branch(self) :
        return nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel,
                            kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                            padding_mode='zeros', stride=self.stride,
                            groups=self.groups, bias=self.use_bias)
        
    def _get_equivalent_kernel_bias(self) :
        kernel_b1_kxk, bias_b1_kxk = self._fuse_conv_bn_module(self.rbr_kxk)
        if self.rbr_1x1 == None :
            kernel_b2_1x1, bias_b2_1x1 = 0, 0 
        else :
            kernel_b2_1x1, bias_b2_1x1 = self._fuse_conv_bn_module(self.rbr_1x1)
            kernel_b2_1x1 = self._pad_kernel(kernel_b2_1x1)
        kernel_b3_id, bias_b3_id = self._fuse_conv_bn_module(self.rbr_identity)

        return self._fuse_addbranch( (kernel_b1_kxk, kernel_b2_1x1, kernel_b3_id), 
                                     (bias_b1_kxk, bias_b2_1x1, bias_b3_id) )
    
    def del_ori_bracnch(self):
        # 消除梯度更新
        [para.detach_() for para in self.parameters()]

        # del no use branch 
        # branch 1
        self.__delattr__('rbr_kxk')
        # branch 2
        self.__delattr__('rbr_1x1')
        # branch 3
        self.__delattr__('rbr_identity')

class RepVGGStage(nn.Module):
    def  __init__(self, in_channels: int, out_channels: int, num_units: int, 
                kernel_size: Union[float, Tuple[float, float]]  = (3, 3), stride: int = 2, dilation: int = 1, group: int=2, 
                fused: bool = False, use_bias: bool = False, use_se: bool = False) :
        """ Build a stage of RepLK model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of block modeules in this stage.
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution. Default: 2
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input channels to output channels. Default: 2
            fused: Whether to be deploy status or training status. Default: False
            use_bias: Whether to use conv bias.
            use_se: Whether to use se. Default: False
        """
        super().__init__()
        strides = [stride] + [1]*(num_units-1)
        dilations = [dilation] * num_units
        blocks = []
        for i in range(num_units):
            cur_group = group if (i % 2) else 1
            blocks.append(RepVGGBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=strides[i], dilation=dilations[i], groups=cur_group, 
                                        fused=fused, use_nonlinear=True, use_bias=use_bias, use_se=use_se))
            in_channels = out_channels
        self.repVGG_stage = nn.ModuleList(blocks)

    def forward(self, x):
        for unit in self.repVGG_stage:
            x = unit(x)
        return x

    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.merge_kernel()

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = [
        [16, (7, 7), 2, 1, 1],
        [32, (5, 7), 2, 3, 4],
        [8, (3, 1), 2, 3, 4],
        [16, (1, 1), 1, 3, 4],
        [64, (3, 3), 1, 1, 1],
    ]
    input=torch.randn(16, 32, 112, 112).to(device)

    print(f"--- Comparing blocks ---")

    for opt in options:
        print(f"Comparing for {opt} ...")

        rep_vgg = RepVGGBlock(input.size(1), out_channels=opt[0], kernel_size=opt[1], 
                              stride=opt[2], dilation=opt[3], groups=opt[4])
        rep_vgg.to(device)
        rep_vgg.eval()
        
        # ori result
        out = rep_vgg(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(rep_vgg, input)
        summary(copy.deepcopy(rep_vgg).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(rep_vgg, input, "ori_model.onnx") 
        
        # fuse result
        rep_vgg.merge_kernel()
        out2 = rep_vgg(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(rep_vgg, input)
        summary(copy.deepcopy(rep_vgg).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(rep_vgg, input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")