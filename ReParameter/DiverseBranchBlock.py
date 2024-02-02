import torch, sys
from torch import nn
from typing import Union, Tuple

sys.path.append("..")
from ReParameter.baseModule import FusedBase
from Conv.Basic.common import ConvBN

__all__ = ["DBBlock"]
# Diverse Branch Block: Building a Convolution as an Inception-like Unit (https://arxiv.org/abs/2103.13425)
# CVPR 2021

class DBBlock(FusedBase):
    """
             |
             |------------------------------------------------
             |   (1)        |  (2)           |  (3)          |  (4)
             |              |          ------------     ------------
             |              |          | 1x1 Conv |     | 1x1 Conv |
             |              |          ------------     ------------
             |              |          |    BN    |     |    BN    |
             |              |          ------------     ------------ 
             |              |                |               |
        ------------    -----------    ------------          |
        | kxk Conv |    | 1x1 Conv|    | kxk Conv |          |
        ------------    -----------    ------------    ------------
        |    BN    |    |   BN    |    |    BN    |    |    Avg   |
        ------------    -----------    ------------    ------------ 
             |               |               |               |
             |------------------------------------------------
             |

                                    |
                                    V

             |
             |------------------------------------------------
             |   (1)        |  (2)           |  (3)          |  (4)
             |              |          ------------     ------------
             |              |          | 1x1 Conv |     | 1x1 Conv |
             |              |          ------------     ------------
             |              |                |               |
        ------------    ------------   ------------     ------------
        | kxk Conv |    | kxk Conv |   | kxk Conv |     | kxk Conv |
        ------------    ------------   ------------     ------------ 
             |               |               |               |
             |------------------------------------------------
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
                        single_init: bool = False):
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
        """
        super(DBBlock, self).__init__(in_channels, out_channels, kernel_size, groups, stride)
        assert dilation==1, "current dilation can't large than 1."
        self.dilation = dilation
        self.padding =  (dilation*(self.kernel_size[0]-1)//2, dilation*(self.kernel_size[1]-1)//2)
        self.activation = nn.ReLU() if use_nonlinear else nn.Identity()
        self.use_bias = use_bias
        self.fused = fused

        if fused:
            self.dbb_reparam = self._fuse_branch()
        else:
            # branch 1
            self.dbb_kxk = ConvBN(in_channels, out_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding, dilation=dilation, groups=groups)    

            # branch 2
            if groups < out_channels:
                self.dbb_1x1 = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

            # branch 3
            self.dbb_1x1_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_1x1_avg.add_module("pw", 
                                            ConvBN(in_channels, out_channels, kernel_size=1, padding=self.padding if use_bias else 0, stride=1, groups=groups, bias=use_bias))
                self.dbb_1x1_avg.add_module('avg', nn.AvgPool2d(kernel_size=self.kernel_size, stride=stride, padding = 0 if use_bias else self.padding ))
            else:
                self.dbb_1x1_avg.add_module('avg', nn.AvgPool2d(kernel_size=self.kernel_size, stride=stride, padding=self.padding))
            self.dbb_1x1_avg.add_module('bnavg', nn.BatchNorm2d(out_channels))

            # branch 4
            self.dbb_1x1_kxk = nn.Sequential()
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels
            self.dbb_1x1_kxk.add_module("pw", 
                                     ConvBN(in_channels, internal_channels_1x1_3x3, kernel_size=1, padding=self.padding if use_bias else 0, stride=1, groups=groups, bias=use_bias))
            self.dbb_1x1_kxk.add_module(f"conv{self.kernel_size[0]}x{self.kernel_size[1]}", 
                                     ConvBN(internal_channels_1x1_3x3, out_channels, kernel_size=self.kernel_size, padding=0 if use_bias else self.padding, stride=stride, dilation=dilation, groups=groups, bias=use_bias))
            
        # The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            # Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self._init_gamma(0.0)
            if hasattr(self, "dbb_kxk"):
                torch.nn.init.constant_(self.dbb_kxk.bn.weight, 1.0)

    def _init_gamma(self, gamma_value):
        if hasattr(self, "dbb_kxk"):
            torch.nn.init.constant_(self.dbb_kxk.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            torch.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_avg"):
            torch.nn.init.constant_(self.dbb_1x1_avg.bnavg.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            torch.nn.init.constant_(self.dbb_1x1_kxk.bnkxk.weight, gamma_value)

    def forward(self, inputs):
        if (self.fused):
            return self.activation(self.dbb_reparam(inputs))

        out = self.dbb_kxk(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_1x1_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.activation(out)
    
    def merge_kernel(self):
        if (self.fused): return 0
        self.fused = True
        kernel, bias = self._get_equivalent_kernel_bias()
        self.dbb_reparam = self._fuse_branch()
        self.dbb_reparam.weight.data = kernel
        if (self.use_bias) :
            self.dbb_reparam.bias.data = bias
        self.del_ori_bracnch()

    def _fuse_branch(self) :
        return nn.Conv2d(in_channels=self.input_channel, out_channels=self.output_channel,
                        kernel_size=self.kernel_size, padding=self.padding, dilation=self.dilation, 
                        padding_mode='zeros', stride=self.stride,
                        groups=self.groups, bias=self.use_bias)
        
    def _get_equivalent_kernel_bias(self):
        # branch 1
        kernel_b1_kxk, bias_b1_kxk = self._fuse_conv_bn_module(self.dbb_kxk)

        # branch 2
        if hasattr(self, 'dbb_1x1'):
            kernel_b2_1x1, bias_b2_1x1 = self._fuse_conv_bn_module(self.dbb_1x1)
            kernel_b2_1x1 = self._pad_kernel(kernel_b2_1x1)
        else:
            kernel_b2_1x1, bias_b2_1x1 = 0, 0

        # branch 3
        kernel_avg, _ = self._fuse_avgpooling(self.dbb_1x1_avg.avg)
        kernel_b3_kxk, bias_b3_kxk = self._fuse_conv_bn(kernel_avg.to(self.dbb_1x1_avg.bnavg.weight.device), self.dbb_1x1_avg.bnavg)
        if hasattr(self.dbb_1x1_avg, 'pw'):
            kernel_b3_1x1, bias_b3_1x1 = self._fuse_conv_bn_module(self.dbb_1x1_avg.pw)
            kernel_b3_1x1_avg, bias_b3_1x1_avg = self._fuse_conv_1x1_kxk(kernel_b3_1x1, bias_b3_1x1, kernel_b3_kxk, bias_b3_kxk)
        else:
            kernel_b3_1x1_avg, bias_b3_1x1_avg = kernel_b3_kxk, bias_b3_kxk

        # branch 4
        kernel_b4_1x1, bias_b4_1x1 = self._fuse_conv_bn_module(self.dbb_1x1_kxk. \
                                                               __getattr__(f"pw")) 
        kernel_b4_kxk, bias_b4_kxk = self._fuse_conv_bn_module(self.dbb_1x1_kxk. \
                                                               __getattr__(f"conv{self.kernel_size[0]}x{self.kernel_size[1]}"))

        kernel_b4_1x1_kxk, bias_b4_1x1_kxk = self._fuse_conv_1x1_kxk(kernel_b4_1x1, bias_b4_1x1, kernel_b4_kxk, bias_b4_kxk)

        return self._fuse_addbranch((kernel_b1_kxk, kernel_b2_1x1, kernel_b3_1x1_avg, kernel_b4_1x1_kxk), 
                                    (bias_b1_kxk, bias_b2_1x1, bias_b3_1x1_avg, bias_b4_1x1_kxk))

    def del_ori_bracnch(self) :
        for para in self.parameters():
            para.detach_()
        self.__delattr__('dbb_kxk')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_avg')
        self.__delattr__('dbb_1x1_kxk')


if __name__ == '__main__':
    from common import save_model_to_onnx, benchmark_model
    from torchsummary import summary
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    options = [
        [64, (7, 7), 1, 1, True, True],
        [16, (7, 3), 2, 1, False, True],
        [16, (5, 7), 2, 4, False, False],
        [8, 3, 2, 4, False, False],
    ]
    input=torch.randn(2, 16, 112, 112).to(device)

    print(f"--- Comparing blocks ---")
    for opt in options:
        print(f"Comparing for {opt} ...")

        dbb = DBBlock(input.size(1),  out_channels=opt[0], kernel_size=opt[1],
                      stride=opt[2], groups=opt[3], use_nonlinear=opt[4], use_bias=opt[5])
        dbb.to(device)
        dbb.eval()

        # ori result
        out = dbb(input)
        ori_time, ori_mem, ori_flops, ori_params = benchmark_model(dbb, input)
        summary(copy.deepcopy(dbb).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(dbb, input, "ori_model.onnx") 

        # fuse result
        dbb.merge_kernel()
        out2 = dbb(input)
        fused_time, fused_mem, fused_flops, fused_params = benchmark_model(dbb, input)
        summary(copy.deepcopy(dbb).to("cpu"), input.shape[1:],  batch_size=-1, device="cpu")
        save_model_to_onnx(dbb.to(device), input, "fused_model.onnx") 

        print('difference parametrization between ori and fuse module')
        print(f"Time   diff: {ori_time:0.2f}s -> {fused_time:0.2f}s")
        print(f"Flops  diff: {ori_flops:s} -> {fused_flops:s}")
        print(f"Params diff: {ori_params:s} -> {fused_params:s}")
        print(f"Memory diff: {ori_mem:s} -> {fused_mem:s}")
        print("Sum diff: ", ((out2 - out) ** 2).sum())
        print("Max diff: ", (out2 - out).abs().max())
        print("\n")