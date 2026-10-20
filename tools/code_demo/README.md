# AIC · UAV Reconstruction Showcase Video

A standalone offline renderer that turns the cached reconstruction from
[`../make_demo`](../make_demo) into a **34 s / 1920×1080 / 30 fps** showcase
video. It does **not** re-run the model: it reads the existing
`city2-demo` point buffer (`points.bin`, 1.5M points), semantic classes,
camera poses and route, then renders deterministic frames.

Output:

```
output/uav_reconstruction_demo.mp4
output/poster.jpg / output/poster_split.jpg
preview/frame_*.jpg        # key frames for quick inspection
frames/raw/                # per-frame 3D render (JPEG)
frames/final/              # composed 1080p frames
```

## What the video shows

| Time | Shot | Content |
|---|---|---|
| 0–3.2 s | Intro | Title + partial point cloud, model/route overview |
| 3.2–12.6 s | Incremental | Drone route progresses, point cloud grows waypoint by waypoint, live point counts / new-point delta / minimap |
| 12.6–14.6 s | Complete | Full 16-waypoint reconstruction, 1.5M points |
| 14.6–22 s | Split | **One continuous point cloud split at a moving vertical line: left RGB, right semantic**, class legend |
| 22–30 s | Objects | **YOLO-style boxes / contours / leader lines / class+confidence labels** on the RGB point cloud |
| 30–34 s | Outro | Final split view + summary, fade out |

The split is done per-fragment in the stage shader (`gl_FragCoord.x`), so the
point cloud is genuinely one cloud: the left half of every point/splat is RGB,
the right half is semantic. It is not two side-by-side copies.

Object labels are derived from the reconstructed semantic point cloud (not
ground-truth 2D annotations): points are projected into the camera, per-class
masks are connected into regions (KMeans for building/tree, connected components
for vehicle/person), and boxes/contours/leader lines are drawn in Python.

## Run

Use the Python env that has Playwright + Edge support (this is the env used for
the web UI validation):

```bash
cd /data/luhangze/aic/code_demo

# quick key-frame preview (no encoding)
/data/luhangze/miniconda3/envs/modelscope/bin/python render_video.py \
    --preview 1.5,7,13.4,20,25.5,32

# full video
/data/luhangze/miniconda3/envs/modelscope/bin/python render_video.py
```

or:

```bash
./run.sh                 # full render
./run.sh --preview 20,25 # preview frames
```

Requirements: `playwright` (Edge channel), `numpy`, `scipy`, `opencv-python`,
`Pillow`, `ffmpeg`. The model/checkpoints are not needed for this renderer.

## Files

```
code_demo/
├── render_video.py     # timeline, camera path, HUD, capture, composite, ffmpeg
├── stage.html          # 1920×1080 Three.js stage (no DOM HUD)
├── stage.js            # point shader (RGB/seg/split), route/drone/grid
├── objects.py          # projected semantic detections (boxes/contours/labels)
├── palette.json        # video semantic palette + class names
├── run.sh
├── output/             # mp4 + poster
├── preview/            # key frames + contact_sheet.jpg
└── frames/             # raw renders and composed frames (regenerable)
```

## Notes / limitations

- The video uses the full 1.5M-point `city2-demo` cache; the 3D pass is rendered
  by headless Edge at 1080p (roughly 0.3–1 s/frame depending on machine load; 1020 frames).
- The object boxes are a **visualisation of the predicted semantic point cloud**,
  not a trained 2D detector. Confidence is the mean semantic probability of the
  region; box IDs/point counts are informational.
- Geometry is the frozen StreamVGGT depth+pose output; facade stretching and
  sparse areas are model/incremental-reconstruction artifacts, not video bugs.
- The semantic palette is a brighter presentation palette (not the raw AIC
  training palette) so the split view remains readable on a dark background.
- Change `--clip city2-lite` for a faster/lighter render, or edit `DURATION` /
  `FPS` at the top of `render_video.py` to retime.
