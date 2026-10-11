# SPARC Phase 0: measurement tooling, rig geometry, sample schema

Phase 0 builds what is needed to choose the backbone and the depth path **on the device** before any training
money is spent. Nothing here is trained: the networks have their deployed shapes and random (BN-calibrated) weights,
which is what latency and TensorRT op support depend on.

| deliverable | where |
|---|---|
| unified training sample (`Sample`, `LabelMask`, `LaneInstance`), `.npz` round trip | `sparc/data/schema.py` |
| stereo rig geometry from any OpenCV stereo calibration: rectified / network intrinsics, FOV, depth precision vs range | `sparc/rig/stereo.py`, `python -m sparc rig` |
| backbone candidates behind one contract (P2..P5) | `sparc/nn/backbone.py` |
| hybrid encoder, grid-sample deformable attention (+ gather fallback), typed-query decoder with agent / lane heads, drivable-area head, depth splat + IPM + BEV decoder, in-network stereo head, engine wrapper | `sparc/nn/` |
| ONNX export with onnxruntime parity and the TensorRT op audit | `sparc/deploy/export.py` |
| bake-off bundle (per candidate and stage, op probes, reference I/O) + standalone device runner + report | `sparc/tools/bakeoff.py`, `sparc/tools/run_on_device.py`, `python -m sparc bakeoff` |
| recording specification for any rig | `recording/protocol.md` |
| tests (35, CPU) | `tests/test_sparc/` |

## Running the bake-off on the Jetson

On any machine with the training requirements (no GPU, no pretrained weights needed):

```bash
python -m sparc bakeoff export --out runs/bakeoff          # ~5 min on 4 CPU cores, ~1.3 GB
```

Copy `runs/bakeoff/` to the Jetson (JetPack 6.x; only `python3`, `numpy` and `trtexec` are used), then:

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks       # MAXN (Super), fixed clocks
python3 runs/bakeoff/run_on_device.py          # FP16: build, time (CUDA graph, 10 s), compare outputs with fp32 reference
python3 runs/bakeoff/run_on_device.py --precision fp32 --only probe_    # optional: op behaviour without FP16
```

Send back `results_fp16.json` (and `results_fp32.json` if run). `python -m sparc bakeoff report --out runs/bakeoff`
turns them into the decision table: backbone ms, rest-of-network ms (full minus backbone), parity, probe verdicts.
Expect 20-60 min in total, dominated by engine builds.

### Decision rules (fixed before seeing the numbers)

1. A candidate qualifies when its `full` FP16 engine builds, passes parity (dense relative error < 5 %, drivable-area
   agreement > 97 %, top-50 box and ID-embedding recall > 80 %), and **full + stereo head <= 28 ms** median GPU time
   (5 ms of the 33 ms budget are reserved for pre-processing, post-processing and tracking on the CPU / GPU).
2. Among qualifying candidates, the Phase-1 short fine-tune on BDD100K (detection mAP + drivable-area mIoU after one
   epoch with pretrained weights) picks the winner; latency breaks ties within 1 mAP.
3. Op verdicts: `probe_splat_scatter_add` must build and pass; if it fails, the splat moves to a ~40-line TensorRT
   plugin (`bev_pool`), and the IPM-only BEV (`bev_splat=False`) is the fallback engine. If `probe_msda_grid_sample` fails
   or is slow, `attn_mode="gather"` (probe `probe_msda_gather`) replaces it.
4. Depth path: D1 (in-network stereo head) if `probe_stereo_head` + the chosen full graph fit rule 1; otherwise D2/D3
   (`docs/sparc/plan.md` section 0).
