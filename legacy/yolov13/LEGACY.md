# Legacy YOLOv13 multi-task (frozen baseline, tag `yolov13-mt-baseline`)

This directory holds the vendored YOLOv13 fork of ultralytics 8.3.63 and the first
`yolov13_multitask` implementation. It is **not** maintained; it exists as ablation
baseline A0. It must run with this directory first on `PYTHONPATH` so its `ultralytics`
shadows the pip package:

    cd legacy/yolov13 && PYTHONPATH=. python -m yolov13_multitask profile ...

Use a separate virtualenv with torch 2.2.x for it; do not mix with the main env.
Known issues are listed in the review in the PR description.
