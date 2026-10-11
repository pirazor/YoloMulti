from pathlib import Path

from sparc.cli import main

CALIB = Path(__file__).parent / "data" / "stereo_calib_example.yaml"


def test_rig_report(capsys):
    assert main(["rig", "--calib", str(CALIB), "--disparity-width", "640", "320", "--baseline", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "rectified baseline | 99.8 mm" in out and "width 640 (640 px)" in out and "zooms in" in out


def test_rig_report_with_alpha(capsys):
    assert main(["rig", "--calib", str(CALIB), "--alpha", "0.5"]) == 0
    assert "zooms in" not in capsys.readouterr().out


def test_profile_table(capsys):
    assert main(["profile", "--backbones", "yolo26s", "--imgsz", "128", "192", "--stereo"]) == 0
    out = capsys.readouterr().out
    assert "| backbone |" in out and "| yolo26s |" in out and "stereo" in out
