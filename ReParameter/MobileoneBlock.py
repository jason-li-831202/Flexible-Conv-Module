import torch, sys
from torch import nn
from typing import Union, Tuple

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN
from Attention import SEModule, coordAttModule, simAModule


__all__ = ["MobileOneBlock", "MobileOneUnit", "MobileOneStage"]
# MobileOne: An Improved One millisecond Mobile Backbone (https://arxiv.org/abs/2206.04040v2)
# CVPR 2022

class MobileOneBlock(FusedBase):
    """
        MobileOneBlock is a basic rep-style block, including training and deploy status

            |
            |----------------------------------
            |   (1)          |  (2)           | (3)
            |                |                |    
            |  x k           |                |    
        ------------    ------------          |    
        | kxk Conv |    | 1x1 Conv |          |      
        ------------    ------------    ------------ 
        |    BN    |    |   BN     |    |    BN    | 
        ------------    ------------    ------------ 
            |                |               |     
           sum               |               |   
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
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), parallel=1,
                 stride=1, dilation=1, groups=1, 
                 fused=False, use_nonlinear=True, use_bias=False, use_se=False):
        """ Initialization of the class.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the conv kernel
            parallel: Number of linear conv branches. 
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            fused: Whether to be deploy status or training status.
            use_nonlinear: Whether to use activate function.
            use_se: Whether to use se.
            use_bias: Whether to use conv bias.
        """
        super(MobileOneBlock, self).__init__(in_channels, out_channels, kernel_size, groups, stride)
        self.padding =  (dilation*(self.kernel_size[0]-1)//2, dilation*(self.kernel_size[1]-1)//2)

        self.dilation = dilation
        self.fused = fused
        self.use_bias = use_bias
        self.activation =  nn.ReLU() if use_nonlinear else nn.Identity()
        self.se = SEModule.SEBlock_Conv(out_channels, out_channels // 16) if use_se else nn.Identity()

        if fused:
            self.rbr_reparam = self._fuse_branch()
        else:
            self.rbr_kxk_parallel = nn.ModuleList([ ConvBN(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, dilation=dilation, padding=self.padding, groups=groups, bias=use_bias) \
                                                    for _ in range(parallel)])
            self.rbr_1x1 = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=use_bias) \
                            if self.kernel_size != (1, 1) else None
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                                if out_channels == in_channels and stride == 1 else None

            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):

        if (self.fused):
            return self.activation(self.se(self.rbr_reparam(inputs)))

        stack_kxk = torch.stack([rbr_d(inputs) for rbr_d in self.rbr_kxk_parallel]) # [(B, C, f_w, f_h)]*k -> (k, B, C, f_w, f_h)
        rbr_kxk_out = stack_kxk.sum(dim=0) # (k, B, C, f_w, f_h) -> (B, C, f_w, f_h)

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
        stack_kxk_tuple = [self._fuse_conv_bn_module(rbr_kxk) for rbr_kxk in self.rbr_kxk_parallel]
        kernel_b1_kxk = torch.stack([x[0] for x in stack_kxk_tuple]).sum(dim=0)
        bias_b1_kxk = torch.stack([x[1] for x in stack_kxk_tuple]).sum(dim=0)

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
        self.__delattr__('rbr_kxk_parallel')
        # branch 2
        self.__delattr__('rbr_1x1')
        # branch 3
        self.__delattr__('rbr_identity')

class MobileOneUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), parallel=1, 
                 stride=1, dilation=1, 
                 fused=False, use_nonlinear=True, use_bias=False, use_se=False):
        """ Initialization of the class.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the conv kernel
            parallel: Number of linear conv branches. 
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            fused: Whether to be deploy status or training status.
            use_nonlinear: Whether to use activate function.
            use_se: Whether to use se.
            use_bias: Whether to use conv bias.
        """
        super(MobileOneUnit, self).__init__()
        self.mobileone_dw = MobileOneBlock(in_channels, in_channels, kernel_size = kernel_size, parallel=parallel, 
                                            stride=stride, dilation=dilation, groups=in_channels, 
                                            fused=fused, use_nonlinear=True, use_bias=use_bias, use_se=use_se)
        self.mobileone_pw = MobileOneBlock(in_channels, out_channels, kernel_size = 1, parallel=parallel, 
                                            stride=1, dilation=dilation, groups=1, 
                                            fused=fused, use_nonlinear=use_nonlinear, use_bias=use_bias, use_se=use_se)
        
    def forward(self, x):
        return self.mobileone_pw(self.mobileone_dw(x))
        
    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, MobileOneBlock):
                module.merge_kernel()

class MobileOneStage(nn.Module):
    def _init_(self, in_channels: int, out_channels: int, num_units: int, num_se_blocks: int, 
               kernel_size: Union[float, Tuple[float, float]]  = (3, 3), parallel: int = 1, 
               dilation: int = 1, fused: bool = False, use_bias: bool = False) :
        """ Build a stage of MobileOne model.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            num_units: Number of unit modeules in this stage.
            num_se_blocks: Number of se attention modeules in this stage.
            kernel_size: Size of the conv kernel
            parallel: Number of linear conv branches. 
            dilation: Spacing between kernel elements. Default: 1
            fused: Whether to be deploy status or training status. Default: False
            use_bias: Whether to use conv bias.
        """
        super(MobileOneStage, self).__init__()
        # Get strides for all layers
        strides = [2] + [1]*(num_units-1)
        units = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_units:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if ix >= (num_units - num_se_blocks):
                use_se = True
            units.append(MobileOneUnit(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                       parallel=parallel, stride=stride, dilation=dilation,
                                       fused=fused, use_nonlinear=True, use_bias=use_bias, use_se=use_se))
            in_channels = out_channels
        self.mobileone_stage = nn.ModuleList(units)
    
    def forward(self, x):
        for unit in self.mobileone_stage:
            x = unit(x)
        return x

    def merge_kernel(self):
        for module in self.modules():
            if isinstance(module, MobileOneUnit):
                module.merge_kernel()

if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # options
    options = [
        [128, 5, 3, 1, 1, True, False],
        [128, (7, 3), 1, 2, 2, False, False],
        [64, (3, 3), 5, 1, 2, True, True],
    ]
    input = torch.randn(2, 32, 224, 224).to(device)

    print(f"--- Comparing blocks ---")
    for opt in options:
        print(f"Comparing for {opt} ...")

        mobileone = MobileOneUnit(input.size(1),  out_channels=opt[0], kernel_size=opt[1], parallel=opt[2], stride=opt[3], dilation=opt[4],
                                   use_nonlinear=opt[5], use_bias=opt[6])
        mobileone.to(device)
        mobileone.eval()

        # ori result
        out = mobileone(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(mobileone, input)
        summary(mobileone.to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(mobileone.to(device), input, "ori_model.onnx") 
        
        # fuse result
        mobileone.merge_kernel()
        out2 = mobileone(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(mobileone, input)
        summary(mobileone.to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(mobileone.to(device), input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")