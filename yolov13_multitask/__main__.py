"""Allow ``python -m yolov13_multitask <cmd> ...`` invocation."""

from yolov13_multitask.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
