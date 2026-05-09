"""Thin wrapper around timm models for the sign classifier."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


def build_model(name: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.1) -> nn.Module:
    """Create a timm classifier with the requested head size."""
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


DEFAULT_MODEL = "mobilenetv3_small_100"
