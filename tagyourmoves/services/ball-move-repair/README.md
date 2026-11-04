# Ball Move Repair Service

Part of **TagYourMoves**. The object detector misses the ball in many frames: it is hidden
behind players, blurred by fast motion, or simply too small. This service fills those gaps. It
receives the detected frames of a video and returns the same frames with an **estimated ball
position** wherever the ball was missing.

It runs as an independent **FastAPI** microservice inside Docker.

---

## Contents

- [How it works](#how-it-works)
- [API](#api)
- [Running the service](#running-the-service)
- [Local scripts: testing and evaluation](#local-scripts-testing-and-evaluation)
- [Project structure](#project-structure)
- [Tuning](#tuning)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## How it works

The estimation is **offline**: it runs after the whole video has been processed, so for every
gap it knows where the ball was *before* and where it reappears *after*. Using both ends gives
much better estimates than a live tracker, which only knows the past.

```
frames (+ optional video)
   │
   ├─ 1. Camera motion            ── only with the video
   ├─ 2. Optical-flow hints       ── only with the video
   ├─ 3. Ball track selection     ── drop detections that aren't the ball
   ├─ 4. Player tracking          ── link player boxes across frames
   ├─ 5. Kalman RTS smoothing     ── smooth path through the detections
   └─ 6. Gap filling              ── possession or Kalman, per gap
   │
repaired frames
```

1. **Camera motion** ([CameraMotion.py](src/repairers/offline/CameraMotion.py)).
   Handheld or panning cameras make the ball appear to move even when it doesn't. Background
   points (with player boxes masked out) are tracked between frames to measure the camera's
   pan, so the ball's motion can be modelled in camera-stabilized coordinates.
2. **Optical-flow hints** ([OpticalFlowTracker.py](src/repairers/offline/OpticalFlowTracker.py)).
   For up to 8 frames after each detection, the ball's pixels are followed with Lucas-Kanade
   optical flow. These hints are weak, and trusted less with every frame.
3. **Ball track selection** ([Offline.py](src/repairers/offline/Offline.py), `_select_ball_track`).
   The detector also fires on heads, orange shorts, rim nets or a spare ball. Detections are
   grouped into tracklets of physically consistent motion, and dynamic programming picks the
   chain of tracklets with the highest total confidence in which every jump is physically
   possible. Everything else is rejected.
4. **Player tracking** (`_PlayerTracks`). Player boxes are linked across frames by overlap
   (Hungarian matching) and smoothed, so "the same player" means something across a gap.
5. **Kalman RTS smoothing** (`_kalman_smooth`). A constant-velocity Kalman filter runs forward
   over the detections and hints, and an RTS smoother runs backward, giving each gap a smooth
   path that respects both of its ends.
6. **Gap filling.** For every gap between two accepted detections:
   - **Possession:** if the same player holds the ball on both sides *and* the ball has been
     moving with that player, the ball follows the player's box.
   - **Otherwise** the Kalman path is used.
   - Gaps longer than `max_gap_frames` (default 45) are left mostly empty. The only exceptions
     are following the holding player for up to 10 frames at each end, and the optical-flow
     hints near the ends when the video was sent.

Estimates are marked with **`confidence: 0.0`**; real detections keep their detector confidence.

---

## API

Interactive documentation is available at `http://localhost:8000/docs` while the service runs.

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

### `POST /repair-ball`

`multipart/form-data` request:

| Field | Type | Required | Description |
|---|---|---|---|
| `frames_data` | **file** (JSON) | yes | List of **every** frame of the video (format below). |
| `video` | file | no | The video the frames came from. Enables camera-motion compensation and optical-flow hints. |
| `frame_width` | int | no | Video width in pixels. Keeps estimates inside the frame when no video is sent. |
| `frame_height` | int | no | Video height in pixels. |
| `max_gap_frames` | int | no | Longest gap filled end to end. Default `45`. |
| `debug` | bool | no | With a video, also saves a rendered video to `mock/debug_outputs/`. Default `false`. |

> `frames_data` must be sent as a **file**, not a text field. Text form fields are capped at
> 1 MB, which a frame list exceeds after about 600 frames.

**Response:** the same list of frames, sorted by `frame_number`, with estimates filled in.

**Errors:**

| Status | When |
|---|---|
| `400` | Invalid JSON, empty list, missing or duplicate `frame_number`s, a video that can't be opened, or a video whose frames don't start at `frame_number` 0. |
| `422` | A required field is missing, or `frames_data` was sent as a text field instead of a file. |

### Frame format

Matches the `Frame`, `Ball`, `Player` and `Rim` contract classes:

```json
[
  {
    "frame_number": 0,
    "timestamp_ms": 0.0,
    "ball": {"confidence": 0.82, "x1": 1032.5, "y1": 441.0, "x2": 1061.0, "y2": 470.2},
    "players": [
      {"confidence": 0.91, "x1": 980.0, "y1": 300.0, "x2": 1100.0, "y2": 620.0,
       "tracking_id": null, "team": null, "shirt_number": null, "has_ball": false}
    ],
    "rims": []
  },
  {
    "frame_number": 1,
    "timestamp_ms": 33.3,
    "ball": null,
    "players": [],
    "rims": []
  }
]
```

Rules:

- **Send every frame, including frames without a ball** (`"ball": null`). The estimator uses
  the position in the list as time, so `frame_number`s must be consecutive.
- **`confidence > 0`** means a real detection. A ball with **`confidence: 0.0`** is treated as
  a weak hint, not a detection.
- **`players` matter.** Without them the service still works, but can't use possession to fill gaps.
- Extra fields (e.g. `has_ball`, `team`) are kept unchanged in the response.
- Coordinates are pixels, `(x1, y1)` top-left and `(x2, y2)` bottom-right.

### Example (Python)

```python
import json
import requests

with open("frames.json") as f:
    frames_json = f.read()

with open("game.mp4", "rb") as video:
    response = requests.post(
        "http://localhost:8000/repair-ball",
        files={
            "frames_data": ("frames.json", frames_json, "application/json"),
            "video": ("game.mp4", video, "video/mp4"),  # optional
        },
        data={"max_gap_frames": 45},
    )

response.raise_for_status()
repaired = response.json()
estimated = [f for f in repaired if f["ball"] and f["ball"]["confidence"] == 0.0]
print(f"{len(estimated)} frames got an estimated ball")
```

### Example (curl)

```bash
curl -X POST http://localhost:8000/repair-ball -F "frames_data=@frames.json;type=application/json" -F "frame_width=1920" -F "frame_height=1080"
```

### With or without the video?

| | Without video | With video |
|---|---|---|
| Speed | Under a second for ~400 frames | ~25–40 s for ~400 frames at 1080p (CPU) |
| Static camera | Good | About the same |
| Moving / handheld camera | Weaker | Better: camera motion is compensated |

---

## Running the service

### Locally

```bash
pip install -r requirements.txt
```
```bash
uvicorn main:app --port 8000
```

### With Docker

```bash
docker build -t tagyourmoves-ball-move-repair .
```
```bash
docker run --rm -p 8000:8000 tagyourmoves-ball-move-repair
```

Debug videos are written inside the container and disappear when it stops. To keep them, mount
the `mock` folder:

```bash
docker run --rm -p 8000:8000 -v "${PWD}/mock:/app/mock" tagyourmoves-ball-move-repair
```

The image is about 840 MB. It uses `opencv-python-headless`, so no GUI libraries are needed.

---

## Local scripts: testing and evaluation

These run the detector themselves, so they need the extra packages in `requirements-dev.txt`
(YOLO / Ultralytics):

```bash
pip install -r requirements-dev.txt
```

Use a separate environment from the service if you can: the dev file installs the full
`opencv-python`, and having it next to `opencv-python-headless` breaks `cv2`.

Put the test videos (`test_vod.mp4`, `test_vod2.mp4`, ...) and the detector weights
(`yolo26s.pt` by default) in `src/repairers/offline/`. Videos are git-ignored.

### `test_offline.py`: full pipeline on a video

```bash
cd src/repairers/offline
```
```bash
python test_offline.py
```

Runs detection, camera motion, optical flow and estimation on `test_vod2.mp4`, and writes
`output_interpolated.mp4` with **green boxes for detections** and **red boxes for estimates**.

Results are cached next to the video (`<video>.<model>-<imgsz>.detections.json` and
`<video>.camera.npy`), so only the first run is slow. Delete the caches after changing the
detector.

### `evaluate_offline.py`: measuring accuracy

There are no hand-labelled frames, so the evaluation builds its own ground truth: it hides
stretches where the detector saw the ball confidently, runs the estimator, and measures how far
the estimates land from the hidden detections.

```bash
python evaluate_offline.py --no-lk --trials 30
```

| Option | Default | Meaning |
|---|---|---|
| `--videos` | `test_vod.mp4 test_vod2.mp4` | Videos to evaluate. |
| `--model` | `yolo26s.pt` | Detector weights. |
| `--imgsz` | `1280` for the default model | Detector input size. |
| `--trials` | `5` | Random trials per video. |
| `--max-gap` | `45` | Longest gap filled. |
| `--no-lk` | off | Skip optical flow. Much faster: it otherwise re-reads the video every trial. |

Output: mean, median and 90th-percentile error in pixels, and the share of hidden frames
estimated within one ball diameter, overall and per gap length.

Keep in mind: hidden frames are ones where the ball was **visible**, so this measures "the
detector missed a visible ball". Real occlusions can't be scored without labels, and optical
flow looks better here than it is when the ball is truly hidden. Errors are also only
comparable between runs **with the same detector**.

### Changing the detector

Any Ultralytics model works (`.pt`, `.onnx`, ...). Classes are matched **by name**:

| Role | Names recognized |
|---|---|
| Ball | `sports ball`, `ball`, `basketball` |
| Rim | `rim`, `hoop`, `basket` |
| Player | `person`, `human`, `player`, `referee`, and names like `red_23` or `refree_1` |

1. Put the weights in `src/repairers/offline/`.
2. Check its class names:
   ```bash
   python -c "from ultralytics import YOLO; print(YOLO('your_model.pt').names)"
   ```
   If they aren't in the table above, add them to `BALL_NAMES`, `RIM_NAMES` or `PLAYER_NAMES`
   in `test_offline.py`.
3. Set `DEFAULT_MODEL` and `DEFAULT_IMGSZ` in `test_offline.py`. ONNX models only accept the
   input size they were exported with. For `.pt` models, 1280 finds small balls much better
   than 640 but is 3–4× slower.

The detector matters more than anything else in this service: estimates can only be as good as
the detections they are anchored to.

---

## Project structure

```
ball-move-repair/
├── main.py                    FastAPI app: endpoints, request parsing and validation
├── Dockerfile                 Service image (python:3.10-slim, uvicorn on port 8000)
├── requirements.txt           Service dependencies
├── requirements-dev.txt       Extra dependencies for the local scripts (YOLO)
├── .dockerignore              Keeps videos, weights and caches out of the image
├── mock/debug_outputs/        Debug videos written when debug=true (git-ignored)
└── src/repairers/offline/
    ├── pipeline.py            repair_frames(): runs the steps below; used by main.py
    ├── Offline.py             Track selection, player tracking, Kalman smoothing, gap filling
    ├── CameraMotion.py        Camera pan estimation and coordinate conversion
    ├── OpticalFlowTracker.py  Short-range Lucas-Kanade ball tracker
    ├── Ball.py, Frame.py,     Contract classes (copies of the shared contract;
    │   Player.py, Rim.py        do not modify)
    ├── test_offline.py        Local end-to-end run with YOLO detection and rendering
    └── evaluate_offline.py    Accuracy evaluation on hidden detections
```

`Ball.py`, `Frame.py`, `Player.py` and `Rim.py` are part of the shared contract between services
and must not be changed. The shared `contracts/` folder at the repository root is still empty on
this branch, so the service uses these copies. When the shared contracts are filled in, only the
imports in `main.py` need to change.

---

## Tuning

The estimator's parameters are constants at the top of
[Offline.py](src/repairers/offline/Offline.py). Distances are in **ball diameters**, so they
scale with resolution and zoom.

| Constant | Value | Meaning |
|---|---|---|
| `MAX_SPEED` | 2.5 | Max plausible ball movement per frame. |
| `MAX_SIZE_RATIO` | 2.0 | Detections whose sizes differ more than this can't be the same ball. |
| `TRACKLET_GAP` | 5 | Max frames between detections grouped into one tracklet. |
| `DETECTION_NOISE` | 0.1 | Assumed position error of a detection. |
| `HINT_NOISE` | 0.1 | Assumed error of an optical-flow hint, growing with each frame since the last detection. |
| `ACCELERATION_NOISE` | 0.15 | How much the Kalman model lets the ball change velocity per frame. |
| `POSSESSION_CONTEXT` | 10 | Frames around a gap used to check whether the ball moves with the player. |
| `PLAYER_MARGIN` | 0.5 | How far outside a player's box the ball can be and still count as held. |
| `ONE_SIDED_FRAMES` | 10 | Frames to follow the holder when only one end of a gap is known. |
| `PLAYER_IOU` | 0.3 | Min box overlap to link a player across frames. |
| `PLAYER_MAX_MISS` | 5 | Frames a player track survives without a detection. |
| `PLAYER_SMOOTHING` | 9 | Moving-average window for player boxes. |

These values were tuned with `evaluate_offline.py` on two test videos. Re-run it after changing them.

---

## Known limitations

- **Static false detections can be accepted as the ball.** Something that looks like a ball and
  never moves, such as a distant rim's net, forms a perfectly consistent track and can pass
  track selection. Estimates then jump between the real ball and that object.
- **One false detection can drop a whole run of real ones.** Track selection accepts or rejects
  tracklets as a whole, so a stray detection grouped at the start or end of a real run can cause
  the entire run to be rejected.
- **Inside a gap the path is smooth.** A bounce, dribble or catch that happens entirely while
  the ball is hidden isn't reproduced.
- **Long gaps stay mostly empty.** Gaps over `max_gap_frames` are only filled near their ends,
  when a player holds the ball or optical flow could follow it.
- **Package versions aren't pinned.** A Docker build may install newer versions than the ones
  tested locally (e.g. OpenCV 5 instead of 4.13), which shifts results slightly.

---

## Troubleshooting

**`422` with "Expected UploadFile".** `frames_data` was sent as a text field. Send it as a file
(see the examples above).

**`400` "frame_number values must be consecutive".** A frame is missing from the list. Include
every frame, with `"ball": null` where there was none.

**Docker build fails with `CERTIFICATE_VERIFY_FAILED`.** Something on your computer intercepts
HTTPS (often antivirus "web shield" or HTTPS scanning, or a company proxy), and the container
doesn't trust its certificate. Either exclude `pypi.org` and `files.pythonhosted.org` from the
scanning, or export the scanner's root certificate as a PEM file and pass it to pip as a build
secret. That needs a copy of the Dockerfile whose `pip install` line is changed to
`RUN --mount=type=secret,id=ca pip install --no-cache-dir --cert /run/secrets/ca -r requirements.txt`:

```bash
docker build -f Dockerfile.ca --secret id=ca,src=root_ca.pem -t tagyourmoves-ball-move-repair .
```

**Docker Desktop shows an error and quits, with "The file cannot be accessed by the system" in
its log.** Leftover socket files in `%LOCALAPPDATA%\Docker\run` (or
`%LOCALAPPDATA%\docker-secrets-engine`) can't be deleted, often because antivirus is
blocking them. Quit Docker, rename those folders (e.g. to `run.stale`) and start Docker again.
Adding the Docker folders as antivirus exclusions prevents it from coming back.

**`test_offline.py` is slow the first time.** Detection runs on every frame (minutes on a CPU),
then the results are cached.
