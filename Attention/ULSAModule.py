import torch
import torch.nn as nn

torch.set_default_tensor_type(torch.cuda.FloatTensor)

# ULSAM: Ultra-Lightweight Subspace Attention Module for Compact Convolutional Neural Networks (https://arxiv.org/abs/2006.15102)
# CVPR 2020
__all__ = ["ULSAMBlock"]

class SubSpace(nn.Module):

    def __init__(self, nin: int) -> None:
        """ Subspace class.
        Args:
            nin: number of input feature volume.
        """
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out

# Ultra-Lightweight Subspace Attention Module
class ULSAMBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, patch_h: int, patch_w: int, num_splits: int) -> None:
        """ Grouped Attention Block having multiple (num_splits) Subspaces.
        Args:
            in_channels: number of input feature volume.
            out_channels: number of output feature volume.
            patch_h: patch height pixels.
            patch_w: patch width pixels.
            num_splits: number of subspaces
        """
        super(ULSAMBlock, self).__init__()

        assert in_channels % num_splits == 0

        self.nin = in_channels
        self.nout = out_channels
        self.h = patch_h
        self.w = patch_w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        return out

