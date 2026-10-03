"""ADAS multi-task perception: detection + drivable area + lane on a YOLO26 backbone.

Built on a pinned pip ``ultralytics`` (>=8.4). Training uses an optional frozen
foundation-model teacher (feature distillation); only the CNN is deployed.
"""

__version__ = "0.2.0"
