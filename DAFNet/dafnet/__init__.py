"""DAFNet: Dynamic Attention Fusion Network for Medical Image Classification.

DAFNet is a lightweight yet powerful CNN architecture designed for medical image
classification tasks. It combines reparameterizable convolutions (MobileOne, RepLKNet),
channel shuffle operations, feature map rotation, and a channel attention mechanism
to achieve strong performance with efficient inference.

Model Variants:
    - DAFNet-Tiny:  dims=[96, 192, 384, 768], depth=[2, 2, 2, 2]   (~4.5M params)
    - DAFNet-Small: dims=[96, 192, 384, 768], depth=[2, 2, 6, 2]   (~8.5M params)
    - DAFNet-Base:  dims=[96, 192, 384, 768], depth=[2, 2, 10, 2]  (~12.5M params)
    - DAFNet-Large: dims=[96, 192, 384, 768], depth=[2, 2, 18, 2]  (~20.0M params)
"""

from .model import (
    DAFNet,
    Stem,
    BaseStage,
    BaseBlock,
    CPD,
    FFEN,
    Attention,
    create_model,
    list_models,
)
from .mobileone import MobileOneBlock, reparameterize_model
from .replknet import ReparamLargeKernelConv

__all__ = [
    "DAFNet",
    "Stem",
    "BaseStage",
    "BaseBlock",
    "CPD",
    "FFEN",
    "Attention",
    "create_model",
    "list_models",
    "MobileOneBlock",
    "reparameterize_model",
    "ReparamLargeKernelConv",
]