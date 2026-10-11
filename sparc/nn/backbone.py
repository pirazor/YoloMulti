"""Backbone candidates for the Phase-0 bake-off, all with one output contract.

``Backbone(x)`` takes an ImageNet-normalised RGB batch (B, 3, H, W) with H, W multiples of 32 and returns
``{"p2": (B, 64, H/4, W/4), "p3": (B, C3, H/8, W/8), "p4": (B, C4, H/16, W/16), "p5": (B, C5, H/32, W/32)}``;
``out_channels`` gives C3..C5. P2 is the high-resolution path for lane / drivable-area detail and for stereo
matching.

Candidates (``BACKBONES``):

| key | trunk | pretrained weights (timm) |
|---|---|---|
| ``vit_t16_dinov3`` | ViT-Tiny/16 distilled from DINOv3 (DEIMv2-class) | ``vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m`` |
| ``vit_s16_dinov3`` | ViT-S/16 DINOv3 | ``vit_small_patch16_dinov3.lvd1689m`` |
| ``vit_s16_dinov2`` | ViT-S DINOv2, patch embedding resampled 14 -> 16 (RF-DETR's recipe) | ``vit_small_patch14_dinov2.lvd142m`` |
| ``convnext_t_dinov3`` | ConvNeXt-T distilled from DINOv3 (pure CNN, TensorRT-safe fallback) | ``convnext_tiny.dinov3_lvd1689m`` |
| ``yolo26s`` | YOLO26s backbone (the current adas_mt trunk), reference row | ``yolo26s.pt`` (Ultralytics) |

ViTs give one stride-16 token grid. Like ViTDet / RF-DETR they are turned into a pyramid from four intermediate
blocks (stride 8 by a 2x transposed conv, 16 as is, 32 by a stride-2 conv), and like DEIMv2's Spatial Tuning
Adapter a light CNN on the image adds stride-4/8 detail that patch-16 tokens cannot hold (thin lane markings,
small signs). CNN trunks have native P2..P5.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from sparc.nn.common import ConvBNAct, ln2d

BACKBONES: Dict[str, dict] = {
    "vit_t16_dinov3": {"kind": "vit", "timm": "vit_tiny_patch16_dinov3_qkvb", "tag": "eupe_lvd1689m"},
    "vit_s16_dinov3": {"kind": "vit", "timm": "vit_small_patch16_dinov3", "tag": "lvd1689m"},
    "vit_s16_dinov2": {"kind": "vit", "timm": "vit_small_patch14_dinov2", "tag": "lvd142m", "patch_size": 16},
    "convnext_t_dinov3": {"kind": "timm_cnn", "timm": "convnext_tiny", "tag": "dinov3_lvd1689m"},
    "yolo26s": {"kind": "yolo", "cfg": "yolo26s.yaml"},
}
P2_CH = 64


class SpatialAdapter(nn.Module):
    """Light CNN on the image: stride-4 (64 ch) and stride-8 (128 ch) detail features (DEIMv2's STA idea)."""

    def __init__(self, c4: int = P2_CH, c8: int = 128):
        super().__init__()
        self.s2 = ConvBNAct(3, 32, 3, 2)
        self.s4 = ConvBNAct(32, c4, 3, 2)
        self.s8 = ConvBNAct(c4, c8, 3, 2)

    def forward(self, x):
        p2 = self.s4(self.s2(x))
        return p2, self.s8(p2)


class ViTBackbone(nn.Module):
    def __init__(self, spec: dict, imgsz: Sequence[int], pretrained: bool):
        super().__init__()
        import timm

        kw = {"img_size": tuple(imgsz)}
        if "patch_size" in spec:
            kw["patch_size"] = spec["patch_size"]  # timm resamples the pretrained patch embedding
        name = spec["timm"] + (f".{spec['tag']}" if pretrained else "")
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0, **kw)
        n, d = len(self.vit.blocks), self.vit.embed_dim
        self.indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        self.adapter = SpatialAdapter()
        c3, c = 192, 256  # the encoder projects everything to 256
        # stride 8: reduce at stride 16 first (cheap), then a 2x transposed conv, then fuse with the adapter detail
        self.up8 = nn.Sequential(ConvBNAct(2 * d, 128, 1), nn.ConvTranspose2d(128, 128, 2, 2), nn.BatchNorm2d(128),
                                 nn.SiLU())
        self.fuse8 = ConvBNAct(128 + 128, c3, 1)
        self.mid16 = ConvBNAct(2 * d, c, 1)
        self.down32 = ConvBNAct(c, c, 3, 2)
        self.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        self.out_channels = [c3, c, c]

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = self.vit.forward_intermediates(x, indices=self.indices, intermediates_only=True, output_fmt="NCHW")
        f = [ln2d(t, ln) for t, ln in zip(feats, self.ln)]
        shallow, deep = torch.cat(f[:2], 1), torch.cat(f[2:], 1)
        p2, a8 = self.adapter(x)
        p4 = self.mid16(deep)
        return {"p2": p2, "p3": self.fuse8(torch.cat([self.up8(shallow), a8], 1)), "p4": p4, "p5": self.down32(p4)}


class TimmCNNBackbone(nn.Module):
    def __init__(self, spec: dict, pretrained: bool):
        super().__init__()
        import timm

        name = spec["timm"] + (f".{spec['tag']}" if pretrained else "")
        self.trunk = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        ch = self.trunk.feature_info.channels()
        self.p2 = ConvBNAct(ch[0], P2_CH, 1)
        self.out_channels = list(ch[1:])

    def forward(self, x):
        c2, c3, c4, c5 = self.trunk(x)
        return {"p2": self.p2(c2), "p3": c3, "p4": c4, "p5": c5}


class YoloBackbone(nn.Module):
    """Layers 0..10 of an Ultralytics YOLO26 model (sequential in the yaml): P2 = layer 2, P3 = 4, P4 = 6, P5 = 10."""

    TAPS = (2, 4, 6, 10)

    def __init__(self, spec: dict, pretrained: bool):
        super().__init__()
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(spec["cfg"], nc=10, verbose=False)
        if pretrained:
            from ultralytics.nn.tasks import load_checkpoint

            src, _ = load_checkpoint(spec["cfg"].replace(".yaml", ".pt"))
            model.load_state_dict(src.float().state_dict(), strict=False)
        self.layers = nn.ModuleList(list(model.model)[: self.TAPS[-1] + 1])
        with torch.no_grad():
            outs = self._run(torch.zeros(1, 3, 64, 64))
        self.p2 = ConvBNAct(outs[0].shape[1], P2_CH, 1)
        self.out_channels = [o.shape[1] for o in outs[1:]]

    def _run(self, x) -> List[torch.Tensor]:
        outs = []
        for i, m in enumerate(self.layers):
            x = m(x)
            if i in self.TAPS:
                outs.append(x)
        return outs

    def forward(self, x):
        p2, p3, p4, p5 = self._run(x)
        return {"p2": self.p2(p2), "p3": p3, "p4": p4, "p5": p5}


def build_backbone(name: str, imgsz: Sequence[int] = (384, 640), pretrained: bool = False) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    if imgsz[0] % 32 or imgsz[1] % 32:
        raise ValueError(f"imgsz {tuple(imgsz)} must be multiples of 32")
    spec = BACKBONES[name]
    if spec["kind"] == "vit":
        return ViTBackbone(spec, imgsz, pretrained)
    if spec["kind"] == "timm_cnn":
        return TimmCNNBackbone(spec, pretrained)
    return YoloBackbone(spec, pretrained)
