"""Frozen DINOv3 teacher (via timm) for training-time feature distillation.

The teacher runs on the training GPU only and never ships: it produces dense patch-token targets for the
student's neck features. Weights are fetched by timm (``pretrained=True``) or loaded from a local
``checkpoint``. Forward cost at 384x640 (GFLOPs): ViT-S 42, S+ 55, B 165, L 584, H+ 1621 (the YOLO26s
training step is ~54), so ViT-B is a balanced default and ViT-L the high-quality option on a big cloud GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIASES = {
    "dinov3_s": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_s_plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_b": "vit_base_patch16_dinov3.lvd1689m",  # default
    "dinov3_l": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_h_plus": "vit_huge_plus_patch16_dinov3.lvd1689m",
}
DEFAULT_TEACHER = "dinov3_b"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenTeacher(nn.Module):
    """``forward(img01) -> (tokens (B, N, D) float32, (gh, gw))`` for RGB images in [0, 1]."""

    def __init__(
        self,
        name: str = DEFAULT_TEACHER,
        pretrained: bool = True,
        checkpoint: str | Path | None = None,
        input_scale: float = 1.0,
        dtype: str = "auto",
    ):
        super().__init__()
        self.name = ALIASES.get(name, name)
        self.model = timm.create_model(
            self.name, pretrained=pretrained and checkpoint is None, num_classes=0, dynamic_img_size=True
        )
        if checkpoint is not None:
            self._load_local(checkpoint)
        self.requires_grad_(False)
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.num_prefix = int(self.model.num_prefix_tokens)
        self.dim = int(self.model.num_features)
        self.input_scale = float(input_scale)
        self.dtype = dtype
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.eval()  # wrapper + timm model; train() below keeps it in eval

    def _load_local(self, path: str | Path) -> None:
        path = str(path)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if len(missing) > 0.1 * len(self.model.state_dict()):
            raise ValueError(f"teacher checkpoint {path} does not match {self.name}: {len(missing)} missing keys")

    def train(self, mode: bool = True):  # always eval
        return super().train(False)

    def grid(self, hw: Tuple[int, int]) -> Tuple[int, int]:
        return (max(1, round(hw[0] * self.input_scale / self.patch)), max(1, round(hw[1] * self.input_scale / self.patch)))

    def _autocast(self, device_type: str):
        if device_type != "cuda" or self.dtype == "float32":
            return torch.autocast(device_type, enabled=False)
        # including_emulation=False: T4/V100 "support" bf16 only through slow emulation
        dt = torch.bfloat16 if (self.dtype == "auto" and torch.cuda.is_bf16_supported(including_emulation=False)) else torch.float16
        if self.dtype == "bfloat16":
            dt = torch.bfloat16
        return torch.autocast("cuda", dtype=dt)

    @torch.no_grad()
    def forward(self, img01: torch.Tensor):
        gh, gw = self.grid(img01.shape[-2:])
        x = img01.float()
        if (gh * self.patch, gw * self.patch) != tuple(x.shape[-2:]):
            x = F.interpolate(x, size=(gh * self.patch, gw * self.patch), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        with self._autocast(x.device.type):
            tokens = self.model.forward_features(x)
        tokens = tokens[:, self.num_prefix :].float()
        assert tokens.shape[1] == gh * gw, (tokens.shape, gh, gw)
        return tokens, (gh, gw)
