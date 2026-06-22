"""
DAFNet core model architecture.

This module implements the DAFNet (Dynamic Attention Fusion Network) architecture,
a lightweight CNN designed for medical image classification. The network features:

- Reparameterizable convolutions for efficient inference
- Channel shuffle with feature map rotation for enhanced feature diversity
- Dynamic kernel fusion from grouped convolution branches
- Channel attention mechanism (AvgPool + MaxPool fusion)
- Configurable depth for different model scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional

from torch.nn import init
from timm.models.layers import trunc_normal_

from .mobileone import MobileOneBlock
from .replknet import ReparamLargeKernelConv


class Stem(nn.Module):
    """Initial stem module that progressively downsamples the input.

    Input: (3, H, W) -> Output: (out_channels, H/4, W/4)

    The stem uses a large-kernel depthwise conv followed by strided
    MobileOne blocks for efficient downsampling.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(Stem, self).__init__()
        self.step1 = ReparamLargeKernelConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=7,
            stride=1,
            groups=in_channels,
            small_kernel=None,
            inference_mode=False,
            activation=nn.GELU(),
        )
        self.step2 = MobileOneBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            use_act=False,
            inference_mode=False,
            num_conv_branches=1,
        )
        self.step3 = MobileOneBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=mid_channels,
            use_act=False,
            inference_mode=False,
            num_conv_branches=1,
        )
        self.step4 = MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=out_channels,
            use_act=False,
            inference_mode=False,
            num_conv_branches=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step4(self.step3(self.step2(self.step1(x))))


class DAFNet(nn.Module):
    """DAFNet: Dynamic Attention Fusion Network.

    The main network architecture consisting of a Stem, multiple BaseStages
    with downsampling between them, and a classification head.

    Args:
        dims: List of channel dimensions for each stage. Default: [96, 192, 384, 768]
        depth: Number of BaseBlocks in each stage. Default: [2, 2, 2, 2]
        num_classes: Number of output classes. Default: 1000
        cls_ratio: Expansion ratio for the classifier head. Default: 4.0
    """

    def __init__(
        self,
        dims: list = None,
        depth: list = None,
        num_classes: int = 1000,
        cls_ratio: float = 4.0,
    ):
        super(DAFNet, self).__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        if depth is None:
            depth = [2, 2, 2, 2]

        self.length = len(depth)
        self.dims = dims
        self.depth = depth

        # Stem: (3, 224, 224) -> (dims[0], 56, 56)
        self.stem = Stem(
            in_channels=3, mid_channels=dims[0] // 2, out_channels=dims[0]
        )

        # Build stages with downsampling between them
        self.stages = nn.ModuleList([
            BaseStage(
                in_channels=dims[i],
                num_layers=depth[i],
                downsample=(
                    CPD(
                        patch_size=7,
                        stride=1,
                        in_channels=dims[i],
                        embed_dim=dims[i + 1],
                    )
                    if i != (self.length - 1)
                    else FFEN(
                        in_channels=dims[i],
                        out_channels=dims[i],
                        stride=1,
                    )
                ),
            )
            for i in range(self.length)
        ])

        # Classifier head
        hidden_dim = int(dims[-1] * cls_ratio)
        self.conv = MobileOneBlock(
            in_channels=dims[-1],
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=dims[-1],
            inference_mode=False,
            num_conv_branches=1,
            use_scale_branch=False,
        )
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = (
            nn.Linear(hidden_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i in range(len(self.depth)):
            x = self.stages[i](x)
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class BaseStage(nn.Module):
    """A stage composed of multiple BaseBlocks followed by a downsampling layer.

    Args:
        in_channels: Number of input channels
        num_layers: Number of BaseBlocks in this stage
        downsample: Downsampling module (CPD or FFEN)
    """

    def __init__(self, in_channels: int, num_layers: int, downsample: nn.Module):
        super(BaseStage, self).__init__()
        self.in_channels = in_channels
        self.downsample = downsample
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            BaseBlock(in_channels=in_channels, groups=4, idx=idx)
            for idx in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)


class BaseBlock(nn.Module):
    """Core building block of DAFNet.

    The BaseBlock performs:
    1. Grouped 1x1 convolution for channel mixing
    2. Channel shuffle into 4 groups
    3. Feature map rotation (4 different orientations) on each group
    4. Depthwise 3x3 convolution on each group (with BN fusion)
    5. Dynamic kernel fusion from the BN-fused weights
    6. Channel attention mechanism
    7. Residual connection with depthwise convolution path

    Args:
        in_channels: Number of input channels
        groups: Number of shuffle groups (default: 4)
        idx: Block index within the stage (determines attention kernel size)
    """

    def __init__(self, in_channels: int, groups: int = 4, idx: int = 0):
        super(BaseBlock, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.sub_channels = in_channels // groups
        self.idx = idx

        # Grouped 1x1 conv for inter-group mixing
        self.groups_conv = nn.Sequential(
            MobileOneBlock(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                inference_mode=False,
                use_act=False,
                num_conv_branches=1,
            )
        )

        # Per-group depthwise 3x3 convs
        self.subs_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.sub_channels, self.sub_channels,
                    kernel_size=3, stride=1, padding=1,
                    groups=self.sub_channels,
                ),
                nn.BatchNorm2d(self.sub_channels),
            )
            for _ in range(self.groups)
        ])

        # Depthwise conv path (parallel to the shuffle path)
        self.dw_conv = nn.Sequential(
            MobileOneBlock(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.in_channels,
                inference_mode=False,
                use_act=True,
                use_scale_branch=False,
                num_conv_branches=1,
            )
        )

        # Final 1x1 conv
        self.last_conv = nn.Sequential(
            MobileOneBlock(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=False,
                use_act=True,
                num_conv_branches=1,
                use_scale_branch=False,
            )
        )

        # Channel attention with dynamic kernel size
        kernel_size = (self.idx * 2) % 8 + 1
        self.attention = Attention(channels=self.in_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Grouped 1x1 conv
        x = self.groups_conv(x)

        # Channel shuffle into 4 groups
        groups_list = self.channel_shuffle(x, self.groups)

        # Apply rotation and depthwise conv on each group
        operations = [0, 1, 2, 3]
        shuffle_sub_list = []
        for i in range(len(groups_list)):
            rotated = self.rotate_feature_map(groups_list[i], operations[i])
            shuffle_sub_list.append(self.subs_conv[i](rotated))

        # Concatenate groups back
        composed_x = torch.cat(shuffle_sub_list, dim=1)

        # Dynamic kernel fusion from BN-fused weights
        weight, bias = self.get_kernel_bias(composed_x)
        weight = torch.cat(weight, dim=0)
        bias = torch.cat(bias, dim=0)
        composed_x = F.conv2d(
            composed_x, weight, bias, padding=1, groups=self.in_channels
        )

        # Attention + residual connection
        composed_x = self.attention(composed_x)
        y = self.last_conv(composed_x + self.dw_conv(x)) + residual
        return y

    def get_kernel_bias(self, x: torch.Tensor):
        """Collect fused kernels and biases from all sub-convolutions."""
        kernel_conv = []
        bias_conv = []
        for ix in range(len(self.subs_conv)):
            _kernel, _bias = self.fuse_bn_tensor(self.subs_conv[ix])
            kernel_conv.append(_kernel)
            bias_conv.append(_bias)
        return kernel_conv, bias_conv

    def fuse_bn_tensor(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse a Conv2d + BatchNorm2d or standalone BatchNorm2d into a single kernel/bias."""
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, 3, 3),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 3 // 2, 3 // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int):
        """Channel shuffle: split channels into `groups` and return as tuple."""
        b, c, h, w = x.size()
        assert c % groups == 0, "channels must be divisible by groups"
        num = c // groups
        x = x.view(b, groups, num, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(b, c, h, w).contiguous()
        return (
            x[:, :num],
            x[:, num:num * 2],
            x[:, num * 2:num * 3],
            x[:, num * 3:],
        )

    @staticmethod
    def rotate_feature_map(feature_map: torch.Tensor, angle: int = 0):
        """Rotate a feature map by the specified angle (0, 1, 2, or 3).

        0: identity
        1: transpose (swap H and W)
        2: flip H and W
        3: flip then transpose
        """
        if angle == 1:
            feature_map = feature_map.permute(0, 1, 3, 2)
        elif angle == 2:
            feature_map = feature_map.flip(dims=(2, 3))
        elif angle == 3:
            feature_map = feature_map.flip(dims=(2, 3))
            feature_map = feature_map.permute(0, 1, 3, 2)
        return feature_map


class CPD(nn.Module):
    """Channel-wise Patch Downsampling.

    Reduces spatial resolution by 2x and doubles (or changes) channels.
    Uses a large-kernel depthwise conv, strided MobileOne, and refinement conv.

    Input: (C, H, W) -> Output: (embed_dim, H/2, W/2)
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        self.conv1 = ReparamLargeKernelConv(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            groups=in_channels,
            small_kernel=None,
            inference_mode=False,
            activation=nn.GELU(),
        )
        self.conv2 = MobileOneBlock(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2,
            groups=embed_dim,
            use_scale_branch=False,
            num_conv_branches=1,
            use_act=False,
        )
        self.conv3 = MobileOneBlock(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dim,
            use_scale_branch=False,
            num_conv_branches=1,
            use_act=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_p = self.conv2(x1)
        x3 = x2 + x_p
        x3 = self.conv3(x3)
        return x3


class Attention(nn.Module):
    """Channel Attention module.

    Fuses AvgPool and MaxPool features through a depthwise conv,
    then applies a learned linear combination to produce channel-wise
    attention weights.

    Args:
        channels: Number of input channels
        kernel_size: Kernel size for the depthwise conv
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.amp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            MobileOneBlock(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=channels,
                num_conv_branches=1,
                inference_mode=False,
                use_act=False,
                use_scale_branch=False,
            )
        )
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(1, 1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # AvgPool path
        y1 = self.conv(x)  # (B, C, H, W)
        y1 = self.aap(y1)  # (B, C, 1, 1)
        y1 = self.linear(y1)
        y1 = y1.squeeze(-1).permute(0, 2, 1)  # (B, 1, C)

        # MaxPool path
        y2 = self.conv(x)  # (B, C, H, W)
        y2 = self.amp(y2)  # (B, C, 1, 1)
        y2 = self.linear(y2)
        y2 = y2.squeeze(-1).permute(0, 2, 1)  # (B, 1, C)

        # Fuse and apply attention
        y = self.sigmoid(y1 + y2)  # (B, 1, C)
        y = y.permute(0, 2, 1).unsqueeze(-1)  # (B, C, 1, 1)

        return x * y.expand_as(x)


class FFEN(nn.Module):
    """Feed-Forward Enhancement Network.

    Used as the final stage's "downsampling" (actually keeps resolution).
    Applies large-kernel conv, dual-path processing (3x3 + 1x1),
    and two feed-forward layers with dropout.

    Args:
        in_channels: Number of input channels
        hidden_channels: Hidden channel dimension (default: same as in_channels)
        out_channels: Number of output channels (default: same as in_channels)
        kernel_size: Large kernel size for the initial conv
        stride: Stride for the initial conv
        drop: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        drop: float = 0.0,
    ):
        super(FFEN, self).__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.conv = ReparamLargeKernelConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            small_kernel=None,
            inference_mode=False,
            activation=nn.GELU(),
        )

        # Dual-path processing
        self.conv1_1 = MobileOneBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            inference_mode=False,
            use_act=True,
            use_scale_branch=True,
            num_conv_branches=1,
            activation=nn.GELU(),
        )
        self.conv1_2 = MobileOneBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=out_channels,
            inference_mode=False,
            use_act=True,
            use_scale_branch=True,
            num_conv_branches=1,
            activation=nn.GELU(),
        )

        # Feed-forward layers
        self.fc1 = MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels,
            inference_mode=False,
            use_act=True,
            use_scale_branch=True,
            num_conv_branches=1,
            activation=nn.GELU(),
        )
        self.fc2 = MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=False,
            use_act=True,
            use_scale_branch=True,
            num_conv_branches=1,
            activation=nn.GELU(),
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x = x_1 + x_2
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Model registry and factory
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "dafnet_tiny": {
        "dims": [96, 192, 384, 768],
        "depth": [2, 2, 2, 2],
        "cls_ratio": 4.0,
    },
    "dafnet_small": {
        "dims": [96, 192, 384, 768],
        "depth": [2, 2, 6, 2],
        "cls_ratio": 4.0,
    },
    "dafnet_base": {
        "dims": [96, 192, 384, 768],
        "depth": [2, 2, 10, 2],
        "cls_ratio": 4.0,
    },
    "dafnet_large": {
        "dims": [96, 192, 384, 768],
        "depth": [2, 2, 18, 2],
        "cls_ratio": 4.0,
    },
}


def create_model(model_name: str = "dafnet_tiny", num_classes: int = 1000) -> DAFNet:
    """Create a DAFNet model by name.

    Args:
        model_name: One of 'dafnet_tiny', 'dafnet_small', 'dafnet_base', 'dafnet_large'
        num_classes: Number of output classes

    Returns:
        DAFNet model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]
    return DAFNet(
        dims=config["dims"],
        depth=config["depth"],
        num_classes=num_classes,
        cls_ratio=config["cls_ratio"],
    )


def list_models() -> list:
    """Return a list of available model names."""
    return list(MODEL_CONFIGS.keys())


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    for name in list_models():
        model = create_model(model_name=name, num_classes=3)
        output = model(x)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output={output.shape}, params={num_params / 1e6:.2f}M")