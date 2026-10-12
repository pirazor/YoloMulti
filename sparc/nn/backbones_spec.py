"""Backbone candidate registry (no torch import: the CLI reads it to build its argument choices)."""

BACKBONES = {
    "vit_t16_dinov3": {"kind": "vit", "timm": "vit_tiny_patch16_dinov3_qkvb", "tag": "eupe_lvd1689m"},
    "vit_s16_dinov3": {"kind": "vit", "timm": "vit_small_patch16_dinov3", "tag": "lvd1689m"},
    "vit_s16_dinov2": {"kind": "vit", "timm": "vit_small_patch14_dinov2", "tag": "lvd142m", "patch_size": 16},
    "convnext_t_dinov3": {"kind": "timm_cnn", "timm": "convnext_tiny", "tag": "dinov3_lvd1689m"},
    "yolo26s": {"kind": "yolo", "cfg": "yolo26s.yaml"},
}
