# basketball-plays

Turn a broadcast basketball video into possessions with court-coordinate trajectories for every
player and the ball, then render them as a 2D court animation or as an overlay on the footage.

Reference game: Michigan at Duke, 2026-02-21 (ESPN game 401817238), first quarter of
[this YouTube video](https://www.youtube.com/watch?v=gSYUO3E3MBY) (video time 0:00 to 35:35).

## Setup

```bash
uv sync                                   # Python 3.11.9, local deps only
echo "ROBOFLOW_API_KEY=..." > .env         # Roboflow Universe models
uvx modal token new                        # once, Modal account
mkdir -p data && uvx yt-dlp -f "bestvideo[height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]" \
  --download-sections "*0:00-35:40" --force-keyframes-at-cuts --merge-output-format mp4 \
  -o "data/duke_michigan_q1.%(ext)s" "https://www.youtube.com/watch?v=gSYUO3E3MBY"
```

## Scripts

1. Trajectories (GPU perception on Modal, then local post-processing):

   ```bash
   uv run python scripts/extract_trajectories.py data/duke_michigan_q1.mp4 --end 2135 --out data/trajectories.jsonl
   # re-run only the local stage after tuning:
   uv run python scripts/extract_trajectories.py data/duke_michigan_q1.mp4 --skip-gpu --out data/trajectories.jsonl
   ```

2. Court animation of the first N possessions:

   ```bash
   uv run python scripts/render_trajectories.py data/trajectories.jsonl --n 5 --out data/trajectories.mp4
   ```

3. Overlay boxes and identities on the real video for the first N possessions:

   ```bash
   uv run python scripts/overlay_video.py data/duke_michigan_q1.mp4 data/trajectories.jsonl --n 5 --out data/overlay.mp4
   ```

## Output format: `trajectories.jsonl`

One JSON object per possession:

```json
{"possession_id": 0, "start_time": 12.3, "end_time": 31.1, "fps": 10,
 "offense_team": "Duke", "attacking_basket": "left",
 "players": [{"track_id": 1012, "team": "Duke", "jersey": "12", "name": "Cameron Boozer",
              "trajectory": [[12.3, 61.2, 24.8], ...],
              "boxes": [[12.3, 512.0, 210.5, 560.2, 340.0], ...]}],
 "ball": [[12.3, 63.0, 25.1], ...]}
```

- `trajectory` rows are `[video_time_s, x_ft, y_ft]` on a 94 x 50 ft NCAA court, x along the
  length, y across, origin at the corner that is top-left in the canonical broadcast view.
- `boxes` rows are `[video_time_s, x1, y1, x2, y2]` in source pixels.
- `jersey` and `name` are best-effort from jersey OCR matched against the ESPN rosters; `null`
  when there were not enough consistent reads.

## How it works

- **Perception** (`modal_app.py`, L4 GPUs, 5-minute chunks in parallel): Roboflow Universe
  RF-DETR player/ball/number detector, 33-point court keypoint model, SmolVLM2 jersey OCR,
  ByteTrack tracking, SigLIP + UMAP + KMeans team clustering. Writes `data/raw/frames_XX.jsonl`.
- **Post-processing** (`basketball_plays/extract.py`): per-frame RANSAC homography to court
  coordinates, invalid-view filtering (replays, close-ups, ads), possession segmentation from
  which half the players occupy, offense team from `player-in-possession` detections, jersey
  voting, teleport removal and smoothing.
- Design notes: `docs/superpowers/specs/2026-09-10-basketball-plays-design.md`.

## Tests

```bash
uv run pytest
```
