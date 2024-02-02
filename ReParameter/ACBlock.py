import torch, sys
from torch import nn
from typing import Union, Tuple

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN
from Attention import SEModule, coordAttModule, simAModule

__all__ = ["ACBlock"]
# ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric (https://arxiv.org/abs/1908.03930)
# ICCV 2019

class ACBlock(FusedBase):
    """
            |
            |----------------------------------
            |   (1)          |  (2)           | (3)
        ------------    ------------    ------------  
        | kxk Conv |    | 1xk Conv |    | kx1 Conv | 
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
    def __init__(self,  in_channels: int, 
                        out_channels: int, 
                        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                        stride: int = 1, 
                        dilation: int = 1, 
                        groups: int = 1, 
                        fused: bool = False,
                        use_nonlinear: bool = True, 
                        use_bias: bool = False,
                        use_se: bool = False):
        """ Initialization of the class.
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the conv kernel
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            fused: Whether to be deploy status or training status.
            use_nonlinear: Whether to use activate function.
            use_bias: Whether to use conv bias.
            use_se: Whether to use se.
        """
        super(ACBlock, self).__init__(in_channels, out_channels, kernel_size, groups, stride)
        self.dilation = dilation
        self.padding =  (dilation*(self.kernel_size[0]-1)//2, dilation*(self.kernel_size[1]-1)//2)

        self.use_bias = use_bias
        self.fused = fused
        self.activation =  nn.ReLU() if use_nonlinear else nn.Identity()
        self.se = SEModule.SEBlock(out_channels, out_channels // 16) if use_se else nn.Identity()

        if fused:
            self.ac_reparam = self._fuse_branch()
        else:
            self.rbr_kxk = ConvBN(in_channels, out_channels, kernel_size=self.kernel_size, 
                                    stride=stride, dilation=dilation, padding=self.padding, groups=groups, bias=use_bias)
            self.rbr_1xk = ConvBN(in_channels, out_channels, kernel_size=(1, self.kernel_size[1]), 
                                    stride=stride, dilation=dilation, padding=(0, self.padding[1]), groups=groups, bias=use_bias)
            self.rbr_kx1 = ConvBN(in_channels, out_channels, kernel_size=(self.kernel_size[0], 1), 
                                    stride=stride, dilation=dilation, padding=(self.padding[0], 0), groups=groups, bias=use_bias)      

    def forward(self, inputs):
        if(self.fused):
            return self.activation(self.se(self.ac_reparam(inputs)))

        return self.activation(self.se(self.rbr_1xk(inputs) + self.rbr_kx1(inputs) + self.rbr_kxk(inputs)))

    def merge_kernel(self):
        if (self.fused): return 0
        self.fused = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.ac_reparam = self._fuse_branch()

        self.ac_reparam.weight.data = kernel
        if (self.use_bias) :
            self.ac_reparam.bias.data = bias
        self.del_ori_bracnch()

    def _fuse_branch(self) :
        return nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel,
                            kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation,
                            padding_mode='zeros', stride=self.stride,
                            groups=self.groups, bias=self.use_bias)
      
    def _get_equivalent_kernel_bias(self):
        kernel_b1_kxk, bias_b1_kxk = self._fuse_conv_bn_module(self.rbr_kxk)
        kernel_b1_1xk, bias_b1_1xk = self._fuse_conv_bn_module(self.rbr_1xk)
        kernel_b1_kx1, bias_b1_kx1 = self._fuse_conv_bn_module(self.rbr_kx1)
        return self._fuse_addbranch( (kernel_b1_kxk, self._pad_kernel(kernel_b1_1xk), self._pad_kernel(kernel_b1_kx1)), 
                                     (bias_b1_kxk, bias_b1_1xk, bias_b1_kx1))
    
    def del_ori_bracnch(self):
        # 消除梯度更新
        [para.detach_() for para in self.parameters()]

        # del no use branch 
        # branch 1
        self.__delattr__('rbr_kxk')
        # branch 2
        self.__delattr__('rbr_1xk')
        # branch 3
        self.__delattr__('rbr_kx1')


if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = [
        [64, (3, 3), 1, 1, 1, True, True],
        [16, (7, 7), 2, 1, 1, False, True],
        [16, (5, 7), 2, 3, 4, False, False],
        [8, 3, 2, 3, 4, False, False],
    ]
    input=torch.randn(16, 64, 224, 224).to(device)

    print(f"--- Comparing blocks ---")
    for opt in options:
        print(f"Comparing for {opt} ...")

        ac = ACBlock(input.size(1), out_channels=opt[0], kernel_size=opt[1], 
                              stride=opt[2], dilation=opt[3], groups=opt[4], use_nonlinear=opt[5], use_bias=opt[6])
        ac.to(device)
        ac.eval()

        out = ac(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(ac, input)
        summary(copy.deepcopy(ac).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(ac, input, "ori_model.onnx") 

        # fuse result
        ac.merge_kernel()
        out2 = ac(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(ac, input)
        summary(copy.deepcopy(ac).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(ac, input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")