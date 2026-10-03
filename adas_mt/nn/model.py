"""Multi-task YOLO26: NMS-free detection + drivable-area + lane segmentation.

Subclasses Ultralytics' ``DetectionModel`` (pinned pip ``ultralytics`` >= 8.4) and keeps the
YOLO26 backbone, neck and end-to-end ``Detect`` head untouched. Two light heads tap the saved
feature maps: backbone P2 (stride 4, layer 2) and the neck P3/P4 that feed Detect.

Forward:
    train -> {"det": {"one2many":..., "one2one":...}, "da": logits, "da_aux": logits, "ll": logits}
    eval  -> {"det": Detect inference output, "da": logits, "ll": logits}   (logits at input res)
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights, intersect_dicts

from .heads import DAHead, LaneHead

P2_LAYER = 2  # first stride-4 stage of the YOLO26 backbone (C3k2 after the P2/4 conv)


class MultiTaskModel(DetectionModel):
    def __init__(
        self,
        cfg: str = "yolo26s.yaml",
        ch: int = 3,
        nc: int | None = None,
        da_classes: int = 3,
        ll_classes: int = 3,
        verbose: bool = False,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        det = self.model[-1]
        self.p3_layer, self.p4_layer = int(det.f[0]), int(det.f[1])
        self.save = sorted(set(self.save) | {P2_LAYER})  # layer 2 is not consumed by the stock graph
        self.da_classes, self.ll_classes = int(da_classes), int(ll_classes)
        self.da_names = ["background", "direct", "alternative"][: self.da_classes]
        self.ll_names = ["background", "solid", "dashed"][: self.ll_classes]

        c2, c3, c4 = self._tap_channels(ch)
        self.da_head = DAHead((c3, c4), self.da_classes)
        self.ll_head = LaneHead((c2, c3, c4), self.ll_classes)
        for m in (self.da_head, self.ll_head):
            initialize_weights(m)
        self.ll_head._init_bias(0.99)  # initialize_weights does not touch it, but be explicit

        if getattr(det, "one2one_cv2", None) is not None:
            det.end2end = True  # inference uses the NMS-free one-to-one branch -> (B, 300, 6)

    # ------------------------------------------------------------------ graph
    def _tap_channels(self, ch: int, size: int = 128) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, taps = self._run(torch.zeros(1, ch, size, size))
        self.train(was_training)
        p2, p3, p4 = taps
        assert p2.shape[-1] == size // 4, f"layer {P2_LAYER} is not stride 4: {tuple(p2.shape)}"
        assert p3.shape[-1] == size // 8 and p4.shape[-1] == size // 16, "neck taps are not P3/P4"
        return p2.shape[1], p3.shape[1], p4.shape[1]

    def _run(self, x: torch.Tensor):
        y: list = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x, (y[P2_LAYER], y[self.p3_layer], y[self.p4_layer])

    def _predict_once(self, x, profile=False, embed=None):  # type: ignore[override]
        if not hasattr(self, "ll_head"):  # stride discovery inside DetectionModel.__init__
            return super()._predict_once(x, profile, embed)
        size = x.shape[-2:]
        det, (p2, p3, p4) = self._run(x)
        da, da_aux = self.da_head(p3, p4, size)
        ll = self.ll_head(p2, p3, p4, size)
        out: Dict[str, object] = {"det": det, "da": da, "ll": ll}
        if da_aux is not None:
            out["da_aux"] = da_aux
        return out

    # ------------------------------------------------------------------- loss
    def init_criterion(self):  # type: ignore[override]
        from .loss import MultiTaskLoss

        return MultiTaskLoss(self)

    # ------------------------------------------------------------------- fuse
    def fuse(self, verbose: bool = True):  # type: ignore[override]
        super().fuse(verbose=verbose)
        for m in list(self.da_head.modules()) + list(self.ll_head.modules()):
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        return self

    # ---------------------------------------------------------------- weights
    def load(self, weights, verbose: bool = True, min_ratio: float = 0.95):  # type: ignore[override]
        """Transfer a YOLO26 (detection or other) checkpoint into backbone + neck + Detect.

        Raises if less than ``min_ratio`` of the backbone/neck parameters transfer, which is the
        silent failure of loading e.g. an ``s`` checkpoint into an ``n`` model.
        """
        if isinstance(weights, dict):
            src = weights.get("ema") or weights["model"]
        else:
            src = weights
        csd = src.float().state_dict() if hasattr(src, "state_dict") else dict(src)
        own = self.state_dict()
        matched = intersect_dicts(csd, own)

        last = len(self.model) - 1
        trunk = [k for k in own if k.startswith("model.") and int(k.split(".")[1]) < last and own[k].numel() > 1]
        total = sum(own[k].numel() for k in trunk)
        got = sum(own[k].numel() for k in trunk if k in matched)
        ratio = got / max(total, 1)
        if ratio < min_ratio:
            raise ValueError(
                f"only {ratio:.1%} of backbone/neck parameters transferred from the checkpoint "
                f"(need >= {min_ratio:.0%}); checkpoint scale/architecture likely differs from this model "
                f"({self.yaml.get('scale')!r})."
            )
        self.load_state_dict(matched, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(matched)}/{len(own)} tensors; backbone+neck {ratio:.1%} (heads init fresh)")
        self.transfer_ratio = ratio
        return ratio


def _ckpt_scale(ckpt) -> str | None:
    m = ckpt.get("ema") or ckpt["model"] if isinstance(ckpt, dict) else ckpt
    yaml = getattr(m, "yaml", None) or {}
    return yaml.get("scale") or None


def build_model(
    scale: str = "s",
    nc: int = 9,
    da_classes: int = 3,
    ll_classes: int = 3,
    weights: str | Path | None = None,
    verbose: bool = False,
) -> MultiTaskModel:
    """Build ``yolo26{scale}``-based multi-task model, optionally from a pretrained YOLO26 checkpoint."""
    ckpt = None
    if weights is not None:
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        got = _ckpt_scale(ckpt)
        if got and got != scale:
            raise ValueError(f"checkpoint {weights} is scale {got!r} but scale {scale!r} was requested")
    model = MultiTaskModel(f"yolo26{scale}.yaml", nc=nc, da_classes=da_classes, ll_classes=ll_classes, verbose=verbose)
    if ckpt is not None:
        model.load(ckpt)
    return model
