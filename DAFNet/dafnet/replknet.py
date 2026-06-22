"""
ReparamLargeKernelConv: A reparameterizable large-kernel convolution from RepLKNet.

This module uses a large-kernel convolution during training, optionally combined
with a small-kernel convolution branch. At inference time, all branches are fused
into a single large-kernel convolution for efficient deployment.

Reference:
    "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs"
    https://arxiv.org/abs/2203.06717

Original implementation:
    https://github.com/DingXiaoH/RepLKNet-pytorch
"""

from typing import Tuple
import torch
import torch.nn as nn

__all__ = ["ReparamLargeKernelConv"]


class ReparamLargeKernelConv(nn.Module):
    """Large-kernel depthwise convolution with optional small-kernel reparameterization.

    During training:
        out = activation(large_kernel_conv(x) + small_kernel_conv(x))

    During inference (after reparameterize()):
        out = activation(fused_large_kernel_conv(x))

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the large convolution kernel
        stride: Convolution stride
        groups: Number of groups for grouped convolution
        small_kernel: Size of the optional small kernel branch (None to disable)
        inference_mode: If True, use single fused conv directly
        activation: Activation function module
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        activation: nn.Module = nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            # Training mode: large kernel + optional small kernel
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        self.activation(out)
        return out

    def get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the fused kernel and bias from all branches."""
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(
                self.small_conv.conv, self.small_conv.bn
            )
            eq_b += small_b
            eq_k += nn.functional.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """Fuse all training branches into a single Conv2d for inference."""
        eq_k, eq_b = self.get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b

        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
        conv: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse a Conv2d weight with a BatchNorm2d into a single kernel and bias."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Create a Conv2d + BatchNorm2d sequential module."""
        mod_list = nn.Sequential()
        mod_list.add_module(
            "conv",
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias=False,
            ),
        )
        mod_list.add_module("bn", nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list