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

## Results on the first half (video 0:00 to 35:35)

`data/trajectories.jsonl` is committed. The GPU stage ran on 8 Modal L4 containers in about
20 minutes wall-clock (roughly $3 of compute including a smoke test); the local stage takes
under a minute and can be re-run with `--skip-gpu` after changing thresholds.

| Metric | Value |
|---|---|
| Sampled frames | 21,350 at 10 fps |
| Frames with a court homography | 74.5% |
| Frames with 6+ players on court (game view) | 59.6% |
| Possessions | 80 (1,411 s of play) |
| Player tracks | 1,291, median 6.8 s, 602 with a jersey number and name |
| Team clustering | cluster 0 = Michigan, cluster 1 = Duke, confirmed by OCR reads |

Known limitations, in order of impact:

- Tracker id switches still fragment players: a typical possession lists 12 to 26 tracks for
  10 players. Stitching links fragments across gaps under 2.5 s and same-frame handoffs; a
  player who is occluded longer, or who is re-acquired with the wrong team cluster, gets a new
  id. Jersey names are the reliable identity key across fragments.
- Possessions are segmented from the half the players occupy, joined across replays up to 12 s.
  Free-throw sequences merge into the surrounding possession, and a long replay can split one.
  ESPN's play-by-play implies about 58 possessions for the half versus 80 detected.
- The ball position is a ground projection of an airborne object and is missing in about half
  of the frames.
- Jersey OCR at 720p misreads similar digits (2 vs 21, 3 vs 23); a number needs two agreeing
  reads on the team's roster, and concurrent duplicates on one team keep only the stronger vote.

## Tests

```bash
uv run pytest
```
