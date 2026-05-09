"""YOLOv13 multi-task model: detection + drivable-area + lane segmentation.

Reuses the upstream YOLOv13 backbone, neck, and Detect head unchanged. Adds two
FPN-fused segmentation decoders (one per task) that tap layers 23/27/31 of the
v13 graph (P3, P4, P5 of the FPN). No upstream files are modified.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel, parse_model, yaml_model_load
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import initialize_weights, intersect_dicts

from .seg_head import FPNFusedSegDecoder

# Layer indices in `ultralytics/cfg/models/v13/yolov13.yaml` that correspond
# to the FPN P3/P4/P5 outputs consumed by Detect.
V13_FPN_LAYERS: Tuple[int, int, int] = (23, 27, 31)
DEFAULT_V13_CFG = "ultralytics/cfg/models/v13/yolov13.yaml"


class YOLOv13MultiTask(DetectionModel):
    """Detection + drivable-area + lane segmentation in one model.

    Forward modes:
        - dict input  -> compute and return loss (inherited from BaseModel.forward)
        - tensor input, training -> dict {det, da, ll} of training-time outputs
        - tensor input, eval     -> dict {det, da, ll} where det matches the
          upstream Detect inference output (tuple of decoded preds and feats)

    Export-mode (set ``self.export_mode = True``) instead returns a 3-tuple
    ``(det, da, ll)`` to be ONNX-friendly.
    """

    def __init__(
        self,
        cfg: str = DEFAULT_V13_CFG,
        ch: int = 3,
        nc: Optional[int] = None,
        da_classes: int = 3,
        ll_classes: int = 3,
        decoder_mid_ch: int = 128,
        verbose: bool = True,
    ) -> None:
        # Build backbone+neck+Detect via the canonical DetectionModel path so we
        # inherit stride-init, weight-init, and parse_model behaviour.
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Discover P3/P4/P5 channel counts by replaying a dry forward of the
        # already-built graph.  This avoids hardcoding scale-dependent channels.
        ch_p3, ch_p4, ch_p5 = self._infer_fpn_channels(ch)

        self.da_classes = int(da_classes)
        self.ll_classes = int(ll_classes)
        self.da_decoder = FPNFusedSegDecoder((ch_p3, ch_p4, ch_p5), self.da_classes, mid_ch=decoder_mid_ch)
        self.ll_decoder = FPNFusedSegDecoder((ch_p3, ch_p4, ch_p5), self.ll_classes, mid_ch=decoder_mid_ch)

        # Used by the multi-task loss; populated by the trainer from cfg.
        self.use_uncertainty_weighting = False
        # Tuple-output toggle for ONNX export.
        self.export_mode = False

        # Ensure newly added decoders have sane init.
        for m in (self.da_decoder, self.ll_decoder):
            initialize_weights(m)

        if verbose:
            LOGGER.info(
                "YOLOv13MultiTask: P3=%d, P4=%d, P5=%d ch | da_classes=%d, ll_classes=%d",
                ch_p3,
                ch_p4,
                ch_p5,
                self.da_classes,
                self.ll_classes,
            )

    # ------------------------------------------------------------------ helpers
    def _infer_fpn_channels(self, ch: int) -> Tuple[int, int, int]:
        """Run a single dry forward over the parsed backbone+neck, recording the
        channel counts at the FPN layers of interest.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, ch, 256, 256)
            captured: Dict[int, int] = {}
            y: list = []
            for m in self.model[:-1]:  # everything except the final Detect
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                x = m(x)
                y.append(x if m.i in self.save else None)
                if m.i in V13_FPN_LAYERS:
                    captured[m.i] = x.shape[1]
        self.train(was_training)
        try:
            return tuple(captured[i] for i in V13_FPN_LAYERS)  # type: ignore[return-value]
        except KeyError as e:  # pragma: no cover - sanity guard
            raise RuntimeError(
                f"FPN layer index {e} not found; this model config likely diverges from v13"
            )

    # --------------------------------------------------------------- predict path
    def _predict_once(self, x, profile: bool = False, visualize: bool = False, embed=None):  # type: ignore[override]
        """Override to capture FPN intermediates and append seg outputs.

        Returns a dict ``{det, da, ll}`` in both training and eval modes. The
        ``det`` value matches the upstream Detect output for the given mode
        (training: list of feats; eval: tuple ``(decoded, feats)`` unless
        Detect.export is set, in which case it is the decoded tensor).

        Some upstream code paths (notably the stride-discovery dry-run inside
        ``DetectionModel.__init__``) call ``self.forward`` *before*
        ``self.da_decoder`` exists; in that case we transparently fall back to
        the parent implementation and return the bare Detect output.
        """
        if not hasattr(self, "da_decoder"):
            return DetectionModel._predict_once(self, x, profile=profile, visualize=visualize, embed=embed)

        y: list = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        det_out = x

        feats = (y[V13_FPN_LAYERS[0]], y[V13_FPN_LAYERS[1]], y[V13_FPN_LAYERS[2]])
        # If save list happens not to contain a FPN index (shouldn't happen with
        # canonical v13.yaml since Detect consumes all three), guard with a
        # clear error.
        if any(f is None for f in feats):
            raise RuntimeError(
                "FPN intermediates not captured. Ensure the model config saves layers 23/27/31."
            )
        da_logits = self.da_decoder(feats)
        ll_logits = self.ll_decoder(feats)

        if self.export_mode:
            return det_out, da_logits, ll_logits
        return {"det": det_out, "da": da_logits, "ll": ll_logits}

    # --------------------------------------------------------------- loss path
    def init_criterion(self):  # type: ignore[override]
        from yolov13_multitask.loss.multitask_loss import MultiTaskLoss

        return MultiTaskLoss(self)

    def loss(self, batch, preds=None):  # type: ignore[override]
        """Compute multi-task loss. ``batch`` is the dict from the dataloader."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self._predict_once(batch["img"])
        return self.criterion(preds, batch)

    # --------------------------------------------------------------- pretrained
    def load(self, weights, verbose: bool = True) -> None:
        """Load a pretrained v13 checkpoint into backbone/neck/detect-head only.

        Uses ``intersect_dicts`` so that shape-mismatched parameters (e.g. the
        Detect head when nc differs) and the new seg decoders are silently
        skipped.
        """
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        own = self.state_dict()
        csd = intersect_dicts(csd, own)
        missing = self.load_state_dict(csd, strict=False)
        if verbose:
            LOGGER.info(
                "Transferred %d/%d items from pretrained weights "
                "(seg decoders initialized fresh)",
                len(csd),
                len(own),
            )
            if getattr(missing, "missing_keys", None):
                LOGGER.debug("missing keys (expected for new heads): %d", len(missing.missing_keys))
