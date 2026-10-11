# Recording specification for SPARC training data (any rig)

What a vehicle rig must record so its drives can train and evaluate SPARC: 3D agent states, depth-grounded BEV maps,
lane order / direction labels and tracking. It is a specification, not tied to one implementation: any recorder
(a ROS 2 graph, a custom C++ recorder, VisionSense v2) that produces the data below works with the Phase 5
converter (`sparc/data/converters/own_drive.py`).

## 1. Sensors

| sensor | minimum | preferred | why |
|---|---|---|---|
| stereo camera pair | hardware-synchronised eyes, 30 fps, >= 1280 px wide per eye, fixed mount | global shutter (e.g. AR0234), 10-15 cm baseline | depth labels and the in-network stereo head; global shutter keeps both eyes geometrically consistent at speed |
| IMU | 100 Hz accel + gyro, raw | 200 Hz+, temperature logged | ego-motion between frames, pitch / roll of the camera |
| GNSS | 5 Hz fix + velocity + quality | 10 Hz, RTK, PPS into the host clock | ego trajectory: lane order from the driven path, self-supervised BEV accumulation |
| vehicle speed | - | OBD-II / CAN at >= 10 Hz | metric scale of ego-motion when GNSS is poor |

Baseline and resolution set how far stereo depth is useful. Check a calibration with
`python -m sparc rig --calib calib.yaml --disparity-width 640 1440`. For the example rig (1440x900 eyes, 99.8 mm,
`tests/test_sparc/data/stereo_calib_example.yaml`), depth error stays below 10 % up to:

| disparity computed at | range with < 10 % error (0.25 px disparity error) |
|---|---|
| full eye width (offline label generation) | ~53 m |
| 640 px (SPARC input width, in-network stereo head) | ~24 m |
| 320 px (a 320-wide stereo engine) | ~12 m |

So offline depth labels must be computed at full resolution, and the BEV grid beyond ~25 m is supervised by
time-accumulated labels (section 7), not by the depth of the same frame.

## 2. Time

* One clock for everything: the host's monotonic clock, exported with every message (ROS time is fine if all
  nodes use the same source). Record GNSS time and PPS offset alongside when available.
* **Camera timestamp = exposure time** (the driver's frame / buffer timestamp, mid-exposure if available), not the
  time the frame was published after processing. Record both if both exist; the converter uses the exposure time.
* **IMU timestamp = sample time** (data-ready interrupt or the FIFO timestamp), at the sensor's output data rate. Do
  not resample to a lower publish rate and do not stamp at publish time.
* **GNSS timestamp = the fix's own time** (mapped to the host clock), never the time of an unrelated sensor.
* Measure the camera-to-host latency once per rig build (blink an LED in view / tap the rig and align the IMU spike
  with image motion) and record it in the calibration file.

## 3. What to record per drive

Raw data, never only derived data: depth, rectification and pose are regenerated offline with better models
and other parameters.

| stream | content | format |
|---|---|---|
| `left/`, `right/` | raw (unrectified) eye images, both eyes of every frame | JPEG quality >= 95 (hardware encoder), or lossless PNG; file name = exposure timestamp in ns |
| `frames.csv` | per frame: timestamp_ns, exposure_us, gain, frame counter, dropped flag | CSV |
| `imu.csv` | timestamp_ns, ax, ay, az (m/s^2), gx, gy, gz (rad/s), temperature | CSV |
| `gnss.csv` | timestamp_ns, gnss_time, lat, lon, alt, fix type, satellites, hdop, speed, track | CSV |
| `speed.csv` | timestamp_ns, speed_mps (optional) | CSV |
| `calib.yaml` | copy of the calibration in force (section 4) with its version id | YAML |
| `drive.yaml` | rig id, calibration version, route, weather, light, driver notes | YAML |

A ROS 2 bag (MCAP, `sensor_msgs/CompressedImage` for the eyes) holding the same information is equally fine;
the converter needs the exposure timestamps, both eyes and the raw IMU / GNSS samples.

Data rate for 2 x 1440x900 JPEG at 30 fps: ~20 MB/s, ~75 GB per hour. Raw RGB would be ~230 MB/s.
Online outputs (depth, detections) can be recorded for debugging, but are never used as labels.

## 4. Calibration (per rig build; again after any mount change)

Stored in one YAML with a version id and date; each drive references the version. Stereo keys follow OpenCV
(`K_left`, `D_left`, `K_right`, `D_right`, `R`, `T`, `R1`, `R2`, `P1`, `P2`, `image_width`, `image_height`), which
`sparc.rig.stereo.StereoRig.from_yaml` reads.

1. **Stereo** (checkerboard, 30-50 well-spread poses at 0.5-2 m, board in every image corner). Accept when per-eye
   RMS < 0.5 px, stereo RMS < 1.0 px, |Tx| within 2 % of the nominal baseline, |Ty|, |Tz| < 5 mm
   (`python -m sparc rig --calib calib.yaml --baseline 0.10` prints the checks).
2. **Rectification field of view.** Rectification with `alpha = 0` keeps only pixels valid in both eyes. On the
   example rig that cuts the horizontal FOV from 78 deg (raw) to 57 deg; `alpha = 0.5` keeps ~83 deg with black
   corners. SPARC's RGB input is the rectified left eye (depth lives in that frame), so choose the alpha for the
   perception FOV you need; `python -m sparc rig --alpha 0.5` reports it. Because the raw eyes are recorded,
   this choice can be changed later.
3. **Camera to ego** (`cam2ego`): mounting height above the ground (tape, +-1 cm, vehicle at nominal load);
   pitch and roll from the IMU's gravity vector at rest on level ground plus the camera-IMU rotation, cross-checked
   with the vanishing point of straight lane markings; yaw relative to the vehicle axis from a straight drive
   (lane vanishing point).
4. **Camera to IMU** rotation: Kalibr camera-IMU calibration, or rotation-only hand-eye from slow rotations in
   front of a checkerboard. Translation by tape measure (not critical).
5. **GNSS antenna** position in the ego frame (tape).
6. **IMU bias**: 2 minutes standing still with the engine running at the start of every drive (recorded as part of
   the drive, flagged in `drive.yaml`).

## 5. Drive plan

* Coverage matrix: urban / suburban / highway / rural x day / dusk / night x dry / rain. Fill the night and rain
  cells deliberately; they are where public datasets are thin.
* Multi-lane roads in both directions, divided and undivided, with opposite traffic (lane order and travel
  direction labels), intersections, merges, construction zones, parked cars, pedestrians and cyclists.
* Drive some routes twice (same direction and reverse): revisits check the self-generated BEV labels for
  consistency.
* Initial volume: ~20 h. Labelling subset: 1 frame per second over 2 h (~7k frames); 500 of them verified by hand for
  lane order and direction (the only set on which those accuracies are reported).

## 6. Acceptance checks per drive

Automated by the Phase 5 converter; a drive that fails is kept but excluded from training until fixed.

* dropped frames < 0.1 %, left / right timestamps identical (hardware sync) or < 1 ms apart;
* IMU: no gap > 2 samples; GNSS: fix on > 95 % of the drive outside tunnels;
* calibration version present; the 2-minute static segment present.

## 7. Offline processing (Phase 5, for reference)

1. Rectify both eyes with the chosen alpha.
2. Depth labels: a large stereo model (FoundationStereo class) on the full-resolution rectified pair, with
   left-right consistency filtering; written as uint16 millimetre PNGs (`Sample.depth`, `depth_source="stereo"`).
3. Ego-motion: GNSS + IMU (+ speed) filter, checked against visual odometry on the rectified pair; written as
   `Sample.world_from_ego` per frame.
4. 2D labels (boxes, lanes, drivable area) from the labelled subset (Supervisely, `adas_mt` converter) and
   pseudo-labels from the trained model elsewhere.
5. **Self-supervised BEV labels by future-near accumulation**: perspective labels lifted to the ground with the depth
   of the frame in which the region is *near* (where stereo is accurate), moved into the ego frame of earlier frames
   with the recorded ego-motion, and accumulated (log-odds). A road region 40 m ahead at time t is labelled by the
   frames, a second or two later, in which it is 5-10 m ahead. This covers the BEV grid beyond stereo range without
   manual BEV annotation.

## 8. Pitfalls seen on an existing rig (avoid in new recorders)

* Images stamped when published after GPU processing instead of at exposure.
* IMU published at half its output data rate, stamped at publish time.
* GNSS fixes stamped with the IMU message time.
* Depth published in the rectified frame while detectors consumed the raw frame: a depth pixel and an image pixel at
  the same coordinates were different scene points (1.5x focal-length difference).
* Only the online, low-resolution depth stored: it cannot serve as a training label beyond ~12 m.
