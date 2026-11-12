# Ball Move Repair Service

Part of **TagYourMoves**. The object detector misses the ball in many frames: it is hidden
behind players, blurred by fast motion, or simply too small. This service fills those gaps. It
receives the detected frames of a video and returns the same frames with an **estimated ball
position** wherever the ball was missing.

It runs as an independent **FastAPI** microservice inside Docker, and works from the frames JSON
alone: no video is sent.

**Contract:** every object in the frames it receives (players, rims, and every ball that is
present) is presumed accurate. Checking detections is the detection service's job; this
service's only job is the missing balls. It never changes a ball it was given. If an input
detection is wrong in practice, that error passes through, and gaps next to it are filled
from it.

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
frames JSON
   │
   ├─ 1. Camera motion from the rims  ── where the camera pans
   ├─ 2. Player tracking              ── link player boxes across frames
   ├─ 3. Kalman RTS smoothing         ── smooth path through the known balls
   └─ 4. Gap filling                  ── possession or Kalman, per gap
   │
the same frames JSON, missing balls filled
```

1. **Camera motion** ([CameraMotion.py](src/repairers/offline/CameraMotion.py)).
   A moving camera makes the ball appear to move even when it doesn't. Rims are fixed to the
   court, so when a rim moves on screen, the camera moved. The same rim is found in consecutive
   frames and its shift gives the camera's pan; frames with no rim in view get an equal share of
   the shift between the frames around them (up to 15 frames). The ball's motion is then
   modelled in camera-stabilized coordinates. With fewer than two frames showing a rim, the
   camera is treated as still.
2. **Player tracking** ([Offline.py](src/repairers/offline/Offline.py), `_PlayerTracks`). When
   every player has a `tracking_id`, those ids are used. Otherwise player boxes are linked
   across frames by overlap (Hungarian matching). Boxes are smoothed, and a track is split where
   the player is missing for more than 5 frames.
3. **Kalman RTS smoothing** (`_kalman_smooth`). A constant-velocity Kalman filter runs forward
   over the known balls, and an RTS smoother runs backward, giving each gap a smooth path that
   respects both of its ends.
4. **Gap filling.** For every gap between two known balls:
   - **Possession:** if the same player holds the ball on both sides *and* the ball has been
     moving with that player, the ball follows the player's box.
   - **Otherwise** the Kalman path is used.
   - Gaps longer than `max_gap_frames` (default 45) are left mostly empty, except for following
     the holding player for up to 10 frames at each end.

Estimates are marked with **`confidence: 0.0`**. Balls that were sent keep their confidence and
coordinates exactly.

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
| `frame_width` | int | no | Video width in pixels. Keeps estimates inside the frame. |
| `frame_height` | int | no | Video height in pixels. |
| `max_gap_frames` | int | no | Longest gap filled end to end. Default `45`. |

> `frames_data` must be sent as a **file**, not a text field. Text form fields are capped at
> 1 MB, which a frame list exceeds after about 600 frames.

**Response:** the frames JSON exactly as it was sent, in the same order and format. Only frames
whose `"ball"` was `null` can change:

| `ball` in the response | Meaning |
|---|---|
| as sent | The frame had a ball; it comes back unchanged. |
| `confidence: 0.0` | The ball was missing and the service estimated it. |
| `null` | The ball was missing and no estimate could be made, e.g. inside a gap longer than `max_gap_frames`. |

**Errors:**

| Status | When |
|---|---|
| `400` | Invalid JSON, empty list, or missing or duplicate `frame_number`s. |
| `422` | `frames_data` is missing, or was sent as a text field instead of a file. |

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
       "tracking_id": 7, "team": null, "shirt_number": null, "has_ball": false}
    ],
    "rims": [{"confidence": 0.88, "x1": 1500.0, "y1": 210.0, "x2": 1560.0, "y2": 240.0}]
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
- **Every ball that is present is used as it is,** whatever its confidence, and never changed.
- **`rims` matter for a moving camera.** They are the only reference for the camera's movement.
  Without rims the camera is treated as still, which is fine for a fixed camera.
- **`players` matter for possession.** Without them gaps are still filled, but only by
  smoothing. A `tracking_id` on every player makes possession more reliable.
- Everything except the filled balls (players, rims, extra fields such as `has_ball` or `team`,
  the frame order) comes back unchanged, so the response can go wherever the request's JSON could.
- Coordinates are pixels, `(x1, y1)` top-left and `(x2, y2)` bottom-right.

### Example (Python)

```python
import requests

with open("frames.json") as f:
    frames_json = f.read()

response = requests.post(
    "http://localhost:8000/repair-ball",
    files={"frames_data": ("frames.json", frames_json, "application/json")},
    data={"frame_width": 1920, "frame_height": 1080},
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

The image needs only plain Python packages (FastAPI, NumPy, SciPy, FilterPy): the service never
reads a video, so it doesn't need OpenCV.

---

## Local scripts: testing and evaluation

### `repair_only.py`: repair a frames file, without the server

The quickest check. It runs only the frame-list repair on a frames JSON file, using the
server's own parsing and response code, so the output is exactly what `POST /repair-ball` would
return. It needs only `requirements.txt`.

```bash
python src/repairers/tests/repair_only.py frames.json 1920 1080
```

Writes `frames.repaired.json` next to the input and prints how many missing balls were filled,
and whether camera motion was used (it needs rims in at least two frames). The width and height
are optional; `--max-gap N` changes the longest gap filled (default 45).

### Scripts that run the detector

The other two scripts run the detector and read the test videos themselves, so they need the
extra packages in `requirements-dev.txt` (YOLO / Ultralytics and OpenCV):

```bash
pip install -r requirements-dev.txt
```

Put the test videos (`test_vod.mp4`, `test_vod2.mp4`, ...) in `src/repairers/tests/` and the
detector weights (`yolo26s.pt` by default) in `moduls/`. Videos are git-ignored. The scripts find
both on their own, so they can be run from any folder; the commands below run from the service root.

### `test_offline.py`: full pipeline on a video

```bash
python src/repairers/tests/test_offline.py
```

Runs detection on `test_vod2.mp4`, then the same repair the service runs, and writes
`output_interpolated.mp4` next to it in `tests/`, with **green boxes for detections** and
**red boxes for estimates**.

Detections are cached next to the video (`<video>.<model>-<imgsz>.detections.json`), so only
the first run is slow. Delete the cache after changing the detector.

### `evaluate_offline.py`: measuring accuracy

There are no hand-labelled frames, so the evaluation builds its own ground truth: it hides
stretches where the detector saw the ball confidently, runs the same repair the service runs,
and measures how far the estimates land from the hidden detections.

```bash
python src/repairers/tests/evaluate_offline.py --trials 30
```

| Option | Default | Meaning |
|---|---|---|
| `--videos` | `test_vod.mp4 test_vod2.mp4` | Videos to evaluate, from `src/repairers/tests/`. |
| `--model` | `yolo26s.pt` | Detector weights, from `moduls/`. |
| `--imgsz` | `1280` for the default model | Detector input size. |
| `--trials` | `5` | Random trials per video. |
| `--max-gap` | `45` | Longest gap filled. |

Output: mean, median and 90th-percentile error in pixels, and the share of hidden frames
estimated within one ball diameter, overall and per gap length.

Keep in mind:

- Hidden frames are ones where the ball was **visible**, so this measures "the detector missed a
  visible ball". Real occlusions can't be scored without labels.
- The test detections are not perfect, unlike the service's contract assumes: false balls
  become anchors, and COCO models such as `yolo26s` find no rims, so camera motion is not used.
  Scores here are therefore a pessimistic estimate.
- Errors are only comparable between runs **with the same detector**.

### Changing the detector

Any Ultralytics model works (`.pt`, `.onnx`, ...). Classes are matched **by name**:

| Role | Names recognized |
|---|---|
| Ball | `sports ball`, `ball`, `basketball` |
| Rim | `rim`, `hoop`, `basket` |
| Player | `person`, `human`, `player`, `referee`, and names like `red_23` or `refree_1` |

1. Put the weights in `moduls/`.
2. Check its class names:
   ```bash
   python -c "from ultralytics import YOLO; print(YOLO('your_model.pt').names)"
   ```
   If they aren't in the table above, add them to `BALL_NAMES`, `RIM_NAMES` or `PLAYER_NAMES`
   in `test_offline.py`.
3. Set `DEFAULT_MODEL` and `DEFAULT_IMGSZ` in `test_offline.py`. ONNX models only accept the
   input size they were exported with. For `.pt` models, 1280 finds small balls much better
   than 640 but is 3–4× slower.

To test camera motion locally, use a model with a rim class: COCO models have none.

---

## Project structure

```
ball-move-repair/
├── main.py                    FastAPI app: endpoints, request parsing and validation
├── Dockerfile                 Service image (python:3.10-slim, uvicorn on port 8000)
├── requirements.txt           Service dependencies
├── requirements-dev.txt       Extra dependencies for the local scripts (YOLO, OpenCV)
├── .dockerignore              Keeps videos, weights and caches out of the image
├── moduls/                    Detector weights for the local scripts (yolo26s.pt, ...)
└── src/repairers/
    ├── contracts/             Contract classes shared between services (do not modify)
    │   └── Ball.py, Frame.py, Player.py, Rim.py
    ├── offline/               The estimation algorithm, used by main.py
    │   ├── pipeline.py            repair_frames(): camera motion, then the estimator
    │   ├── CameraMotion.py        Camera pan from the rims, and coordinate conversion
    │   └── Offline.py             Player tracking, Kalman smoothing, gap filling
    └── tests/                 Local scripts and test videos (not used by the service)
        ├── repair_only.py         Repair a frames JSON file without the server
        ├── test_offline.py        End-to-end run with YOLO detection and rendering
        └── evaluate_offline.py    Accuracy evaluation on hidden detections
```

`Ball.py`, `Frame.py`, `Player.py` and `Rim.py` are part of the shared contract between services
and must not be changed. The shared `contracts/` folder at the repository root is still empty on
this branch, so the service uses its own copies in `src/repairers/contracts/`.

The code imports these modules by their bare names (for example `Frame.py` does
`from Ball import Ball`), so every entry point puts `contracts/` and `offline/` on Python's
import path itself: `main.py`, `test_offline.py` and `evaluate_offline.py`. Those few lines at
the top of each file are the only place to change if the folders move again.

---

## Tuning

The estimator's parameters are constants at the top of
[Offline.py](src/repairers/offline/Offline.py) and
[CameraMotion.py](src/repairers/offline/CameraMotion.py). Distances in `Offline.py` are in
**ball diameters**, so they scale with resolution and zoom.

| Constant | File | Value | Meaning |
|---|---|---|---|
| `MAX_SPEED` | Offline | 2.5 | Max plausible ball movement per frame; sets how unsure the Kalman starts. |
| `DETECTION_NOISE` | Offline | 0.1 | Assumed position error of a given ball (small: the input is presumed accurate). |
| `ACCELERATION_NOISE` | Offline | 0.15 | How much the Kalman model lets the ball change velocity per frame. |
| `POSSESSION_CONTEXT` | Offline | 10 | Frames around a gap used to check whether the ball moves with the player. |
| `PLAYER_MARGIN` | Offline | 0.5 | How far outside a player's box the ball can be and still count as held. |
| `ONE_SIDED_FRAMES` | Offline | 10 | Frames to follow the holder when only one end of a gap is known. |
| `PLAYER_IOU` | Offline | 0.3 | Min box overlap to link a player across frames (when there are no tracking ids). |
| `PLAYER_MAX_MISS` | Offline | 5 | Frames a player may be missing before their track is split. |
| `PLAYER_SMOOTHING` | Offline | 9 | Moving-average window for player boxes. |
| `MAX_RIM_SHIFT` | CameraMotion | 1.5 | Max shift of a rim between two looks at it, in rim widths per frame. |
| `MAX_BRIDGE_FRAMES` | CameraMotion | 15 | Longest stretch without a rim in view that is bridged. |

Re-run `evaluate_offline.py` after changing them.

---

## Known limitations

- **Errors in the input pass through.** By contract, every ball that is sent is trusted. A false
  detection is returned unchanged, and the gaps next to it are filled toward it.
- **Camera motion needs rims in view.** With no rim visible for more than 15 frames, that stretch
  is treated as a still camera. Only the pan is measured, not zoom.
- **Inside a gap the path is smooth.** A bounce, dribble or catch that happens entirely while
  the ball is hidden isn't reproduced.
- **Long gaps stay mostly empty.** Gaps over `max_gap_frames` are only filled near their ends,
  when a player holds the ball.
- **Package versions aren't pinned.** A Docker build may install newer versions than the ones
  tested locally, which can shift results slightly.

---

## Troubleshooting

**`422` with "Expected UploadFile".** `frames_data` was sent as a text field. Send it as a file
(see the examples above).

**`400` "frame_number values must be consecutive".** A frame is missing from the list. Include
every frame, with `"ball": null` where there was none.

**`test_offline.py` is slow the first time.** Detection runs on every frame (minutes on a CPU),
then the results are cached.
