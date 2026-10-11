"""Stereo rig geometry: calibration, rectification, network intrinsics and depth precision.

Input is a stereo calibration YAML in the OpenCV ``stereoCalibrate`` / ``stereoRectify`` layout (any rig):
``image_width``, ``image_height``, ``K_left``, ``D_left``, ``K_right``, ``D_right``, ``R``, ``T`` (right-from-left,
metres), ``R1``, ``R2``, ``P1``, ``P2``; matrices either as ``{rows, cols, data}`` or as nested lists; optional
``reprojection_rms_<name>`` entries. ``tests/test_sparc/data/stereo_calib_example.yaml`` is a real example
(1440x900 eyes, ~100 mm baseline).

Two facts matter for SPARC and are easy to get wrong:

* Stereo depth lives in the **rectified** left frame (pinhole ``P1``), not in the raw left image (``K_left`` +
  distortion). When rectification zooms in (the example: 1.5x), a raw pixel and the same depth pixel are different
  scene points. SPARC therefore takes the rectified left image as its RGB input, so image, depth and the right eye
  share one pinhole ``K`` (:meth:`StereoRig.rectify_maps` builds the maps; :meth:`StereoRig.with_alpha` trades
  valid area for field of view).
* Depth precision falls with the square of range: ``dZ = Z^2 / (f * B) * dd`` for a disparity error ``dd``.
  :meth:`StereoRig.depth_table` gives the numbers at the resolution the disparity is actually computed at; they
  decide how far measured depth can supervise the BEV head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


def _mat(node, rows: Optional[int] = None, cols: Optional[int] = None) -> np.ndarray:
    if isinstance(node, dict) and "data" in node:
        a = np.asarray(node["data"], dtype=np.float64).reshape(int(node["rows"]), int(node["cols"]))
    else:
        a = np.asarray(node, dtype=np.float64)
    if rows is not None:
        a = a.reshape(rows, -1) if cols is None else a.reshape(rows, cols)
    return a


@dataclass(frozen=True)
class StereoRig:
    eye_hw: Tuple[int, int]      # raw eye image size the calibration is for (H, W)
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    R: np.ndarray                # right-from-left rotation
    T: np.ndarray                # right-from-left translation, metres
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    rms: Dict[str, float] = None  # reprojection RMS reported by the calibration, if present

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_yaml(cls, path: str | Path) -> "StereoRig":
        with open(path, "r", encoding="utf-8") as f:
            c = yaml.safe_load(f)
        missing = [k for k in ("image_width", "image_height", "K_left", "D_left", "K_right", "D_right", "R", "T",
                               "R1", "R2", "P1", "P2") if k not in c]
        if missing:
            raise ValueError(f"{path}: not a stereo calibration (missing {missing})")
        rms = {k.replace("reprojection_rms_", ""): float(v) for k, v in c.items() if k.startswith("reprojection_rms_")}
        return cls(
            eye_hw=(int(c["image_height"]), int(c["image_width"])),
            K_left=_mat(c["K_left"], 3, 3), D_left=_mat(c["D_left"]).reshape(-1),
            K_right=_mat(c["K_right"], 3, 3), D_right=_mat(c["D_right"]).reshape(-1),
            R=_mat(c["R"], 3, 3), T=_mat(c["T"]).reshape(3), R1=_mat(c["R1"], 3, 3), R2=_mat(c["R2"], 3, 3),
            P1=_mat(c["P1"], 3, 4), P2=_mat(c["P2"], 3, 4), rms=rms,
        )

    def with_alpha(self, alpha: float) -> "StereoRig":
        """Re-run ``cv2.stereoRectify`` with free-scaling ``alpha`` (0 = only valid pixels, 1 = keep all source
        pixels, black borders included). A larger alpha gives the perception network back field of view at
        the cost of invalid border pixels."""
        import cv2

        h, w = self.eye_hw
        R1, R2, P1, P2, *_ = cv2.stereoRectify(self.K_left, self.D_left, self.K_right, self.D_right, (w, h),
                                               self.R, self.T.reshape(3, 1), alpha=float(alpha),
                                               flags=cv2.CALIB_ZERO_DISPARITY)
        return replace(self, R1=R1, R2=R2, P1=P1, P2=P2)

    # ------------------------------------------------------------------ rectified geometry
    @property
    def K_rect(self) -> np.ndarray:
        return self.P1[:, :3].copy()

    @property
    def baseline_m(self) -> float:
        """Rectified baseline: ``-P2[0,3] / P2[0,0]`` (what the depth conversion uses)."""
        return float(-self.P2[0, 3] / self.P2[0, 0])

    @property
    def bf(self) -> float:
        """``f * B`` in eye pixels x metres: depth = bf / disparity_px."""
        return float(self.P1[0, 0] * self.baseline_m)

    def network_K(self, net_hw: Sequence[int], src_hw: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, Dict]:
        """Intrinsics of the rectified left image after the deployed resize + letterbox to ``net_hw``.

        Uses the exact arithmetic of :func:`adas_mt.deploy.preprocess.letterbox_params` (long side to max(net_hw),
        then centred padding)."""
        from adas_mt.deploy.preprocess import letterbox_params

        src_hw = tuple(src_hw or self.eye_hw)
        _, (uw, uh), top, left, _ = letterbox_params(src_hw, net_hw)
        sx, sy = uw / src_hw[1], uh / src_hw[0]
        K = self.K_rect
        K[0, 0] *= sx
        K[0, 2] = K[0, 2] * sx + left
        K[1, 1] *= sy
        K[1, 2] = K[1, 2] * sy + top
        return K, {"scale_xy": (sx, sy), "top": top, "left": left, "unpad_wh": (uw, uh)}

    def rectify_maps(self, out_hw: Sequence[int], eye: str = "left") -> Tuple[np.ndarray, np.ndarray]:
        """``cv2.remap`` maps from the raw eye image to the rectified image at ``out_hw`` (undistort + rectify +
        resize in one lookup; a CUDA remap kernel on the device does the same with these maps)."""
        import cv2

        h, w = self.eye_hw
        oh, ow = int(out_hw[0]), int(out_hw[1])
        K, D, Rr, P = ((self.K_left, self.D_left, self.R1, self.P1) if eye == "left" else
                       (self.K_right, self.D_right, self.R2, self.P2))
        Kn = P[:, :3].copy()
        Kn[0] *= ow / w
        Kn[1] *= oh / h
        return cv2.initUndistortRectifyMap(K, D, Rr, Kn, (ow, oh), cv2.CV_32FC1)

    # ------------------------------------------------------------------ field of view
    def raw_fov_deg(self) -> Tuple[float, float]:
        """Horizontal / vertical FOV of the raw left eye, distortion included (edge midpoints undistorted)."""
        import cv2

        h, w = self.eye_hw
        pts = np.array([[[0, h / 2]], [[w - 1, h / 2]], [[w / 2, 0]], [[w / 2, h - 1]]], dtype=np.float64)
        n = cv2.undistortPoints(pts, self.K_left, self.D_left).reshape(-1, 2)
        hf = math.degrees(math.atan(-n[0, 0]) + math.atan(n[1, 0]))
        vf = math.degrees(math.atan(-n[2, 1]) + math.atan(n[3, 1]))
        return hf, vf

    def rect_fov_deg(self) -> Tuple[float, float]:
        h, w = self.eye_hw
        f, cx, fy, cy = self.P1[0, 0], self.P1[0, 2], self.P1[1, 1], self.P1[1, 2]
        return (math.degrees(math.atan(cx / f) + math.atan((w - cx) / f)),
                math.degrees(math.atan(cy / fy) + math.atan((h - cy) / fy)))

    # ------------------------------------------------------------------ depth precision
    def depth_table(self, disparity_width: int, ranges_m: Sequence[float] = (5, 10, 15, 20, 30, 40),
                    subpixel: Sequence[float] = (0.1, 0.25, 0.5), max_disp: Optional[int] = None) -> Dict:
        """Depth error ``Z^2 / (f B) * dd`` when disparity is estimated on an image ``disparity_width`` pixels wide
        (the rectified eye resized to that width), for disparity errors ``subpixel`` in those pixels.

        Returns ``{"bf": f*B at that width, "min_depth_m": for max_disp, "rows": [{"range_m", "disp_px",
        "err_m": {dd: metres}}]}``."""
        bf = self.bf * disparity_width / self.eye_hw[1]
        rows = []
        for z in ranges_m:
            rows.append({"range_m": float(z), "disp_px": bf / z,
                         "err_m": {float(dd): z * z / bf * dd for dd in subpixel}})
        return {"disparity_width": int(disparity_width), "bf": bf,
                "min_depth_m": (bf / max_disp) if max_disp else None, "rows": rows}

    def max_range_for(self, disparity_width: int, rel_err: float = 0.1, subpixel: float = 0.25) -> float:
        """Largest range whose depth error stays below ``rel_err * Z``: Z = rel_err * f B / dd."""
        return rel_err * self.bf * disparity_width / self.eye_hw[1] / subpixel

    # ------------------------------------------------------------------ checks
    def check(self, expected_baseline_m: Optional[float] = None) -> List[str]:
        """Human-readable warnings about the calibration (empty list = nothing suspicious)."""
        out = []
        tx, ty, tz = (float(v) for v in self.T)
        if abs(ty) > 0.005 or abs(tz) > 0.005:
            out.append(f"|Ty| or |Tz| > 5 mm (T = {tx * 1e3:.1f}, {ty * 1e3:.1f}, {tz * 1e3:.1f} mm): rectification "
                       "may be distorted; recapture with more pose diversity")
        if expected_baseline_m and abs(abs(tx) - expected_baseline_m) > 0.2 * expected_baseline_m:
            out.append(f"|Tx| = {abs(tx) * 1e3:.1f} mm, expected ~{expected_baseline_m * 1e3:.0f} mm "
                       "(wrong checkerboard square size?)")
        zoom = self.P1[0, 0] / self.K_left[0, 0]
        if zoom > 1.2:
            hr, _ = self.raw_fov_deg()
            hc, _ = self.rect_fov_deg()
            out.append(f"rectification zooms in {zoom:.2f}x: horizontal FOV {hr:.0f} deg raw -> {hc:.0f} deg rectified. "
                       "The perception input must be the rectified image (depth is in that frame); consider "
                       "StereoRig.with_alpha() to keep more field of view")
        for k, v in (self.rms or {}).items():
            lim = 1.0 if k == "stereo" else 0.5
            if v > lim:
                out.append(f"reprojection RMS {k} = {v:.2f} px > {lim} px")
        return out

    def report(self, net_hw: Sequence[int] = (384, 640), runtime_widths: Dict[str, int] | None = None) -> str:
        """Markdown summary used by ``python -m sparc rig``."""
        runtime_widths = runtime_widths or {}
        h, w = self.eye_hw
        Kn, lb = self.network_K(net_hw)
        hr, vr = self.raw_fov_deg()
        hc, vc = self.rect_fov_deg()
        lines = [
            f"| quantity | value |", "|---|---|",
            f"| eye image | {w}x{h} |",
            f"| rectified baseline | {self.baseline_m * 1e3:.1f} mm |",
            f"| fx raw / rectified | {self.K_left[0, 0]:.0f} / {self.P1[0, 0]:.0f} px |",
            f"| FOV raw (h x v) | {hr:.1f} x {vr:.1f} deg |",
            f"| FOV rectified (h x v) | {hc:.1f} x {vc:.1f} deg |",
            f"| network input {net_hw[1]}x{net_hw[0]} | image {lb['unpad_wh'][0]}x{lb['unpad_wh'][1]}, pad top {lb['top']} left {lb['left']}, fx {Kn[0, 0]:.1f} px |",
            "",
            "Depth error (m) for a disparity error of 0.1 / 0.25 / 0.5 px at the width the disparity is computed at:",
            "",
            "| disparity computed at | 5 m | 10 m | 20 m | 30 m | 40 m | range with <10% error at 0.25 px |",
            "|---|---|---|---|---|---|---|",
        ]
        widths = {"full eye": w, **runtime_widths}
        for name, dw in widths.items():
            t = self.depth_table(dw, ranges_m=(5, 10, 20, 30, 40))
            cells = ["{:.2f} / {:.2f} / {:.2f}".format(*r["err_m"].values()) for r in t["rows"]]
            lines.append(f"| {name} ({dw} px) | " + " | ".join(cells) + f" | {self.max_range_for(dw):.0f} m |")
        warn = self.check()
        if warn:
            lines += ["", "Warnings:", ""] + [f"- {x}" for x in warn]
        return "\n".join(lines)
