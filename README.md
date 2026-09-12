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

   Both renderers accept `--n 0` for all possessions and `--split-dir data/clips/...` to also write
   one clip per possession, named `p<id>_<video start>s_<offense>_<outcome>.mp4`. Play-by-play
   events appear as banners at the moment they happen (made or missed shot, foul, turnover,
   rebound), with a coloured ring or box around the player named in the event when that player
   was identified.

4. Attach game clock, score and play-by-play outcome to every possession (needs `tesseract` on the
   PATH, `brew install tesseract`; the ESPN summary is fetched once and cached in `data/`):

   ```bash
   uv run python scripts/annotate_outcomes.py data/duke_michigan_q1.mp4 data/trajectories.jsonl
   # reuse the scoreboard OCR from a previous run:
   uv run python scripts/annotate_outcomes.py data/duke_michigan_q1.mp4 data/trajectories.jsonl --skip-ocr
   ```

5. A CSV for reviewing cases, one row per possession with clock, score, outcome, events, named
   players and the clip file name:

   ```bash
   uv run python scripts/summarize_possessions.py data/trajectories.jsonl --out data/possessions.csv
   ```

6. Play-recognition spike (Phase 0 of `docs/superpowers/specs/2026-09-11-duke-play-recognition-design.md`):
   Duke half-court possessions from ESPN intervals, setup frames and a montage of setup snapshots.

   ```bash
   uv run python scripts/spike_setups.py data/trajectories.jsonl --out data/plays/spike \
       --sensitivity
   ```

   Writes `halfcourt.jsonl`, `setups.png` and `report.md` with the go/no-go numbers.
   `--sensitivity` appends a setup-rate table over stillness thresholds (max move in feet by
   minimum players), which is how the 1 ft spec value gets checked against measured jitter.

## Many games

Games beyond the reference one live in `games.json` (`espn_id`, `youtube_id`, `date`, `opponent`,
`home`, `broadcaster`, `layout`, `split`, `note`) and are ingested one at a time into their own
directory under `data/games/<espn_id>/` (gitignored):

```
data/games/<espn_id>/
  video.mp4            downloaded broadcast (yt-dlp, 720p avc1)
  espn_summary.json    cached ESPN summary (boxscore, play-by-play)
  raw/frames_XX.jsonl  Stage A (GPU) output, one file per 5-minute chunk
  trajectories.jsonl   Stage B output: possessions with court-coordinate trajectories
  scoreboard_raw.jsonl OCR'd clock/score reads, one per second
  halfcourt.jsonl      Duke half-court possession records, one per ESPN interval
  coverage.json        per-period counts: intervals, located, dead-ball, dead-ball-with-setup
  game.json            the registry entry, written after a full run
```

`scripts/ingest_game.py <espn_id>` runs the seven steps (`download`, `espn`, `gpu`, `extract`,
`ocr`, `annotate`, `halfcourt`) in order, skipping any step whose output already exists so a
failed run can be fixed and the same command re-run to resume:

```bash
uv run python scripts/ingest_game.py 401817238 --dry-run     # print the plan, run nothing
uv run python scripts/ingest_game.py 401817238               # run every step, skip existing outputs
uv run python scripts/ingest_game.py 401817238 --steps gpu    # run one step
uv run python scripts/ingest_game.py 401817238 --force halfcourt --steps halfcourt  # redo a step
uv run python scripts/ingest_game.py 401817238 --max-cost 5   # lower the GPU cost ceiling (10)
```

The `gpu` step prints an estimated cost (video minutes x $0.09) before doing anything remote and
refuses to proceed above `--max-cost`.

Each game's broadcast uses one of five scoreboard graphic layouts
(`basketball_plays/broadcasts.py`), set per game in `games.json` and picked automatically by the
`ocr`/`annotate` steps: `espn` (ESPN, ESPN2, ACC Network), `cbs`, `ncaa` (NCAA tournament), `cw`
(The CW) and `cbssn` (CBS Sports Network); each has a fixture frame under
`tests/fixtures/scoreboards/`. Check a layout's regions against a real frame with:

```bash
uv run python scripts/probe_scoreboard.py data/games/<espn_id>/video.mp4 --t 900 --layout cbs
```

### Smoke test: the whole Michigan game (401817238)

`uv run python scripts/ingest_game.py 401817238` end to end on the full 77.7-minute broadcast
(Phase 0 only covered the first half): download (1.98 GB), ESPN summary, GPU stage (16 chunks,
46,618 frames at 10 fps; cost estimate `77.7 video minutes x $0.09 = $6.99`, actual Modal billing
$5.00), extraction (157 possessions), scoreboard OCR (4,634 reads), per-period annotation and
half-court records:

| Period | Span (video s) | Intervals | Located | Dead-ball | With setup |
|---|---|---|---|---|---|
| 1 | 10.3 - 2198.3 | 36 | 32 | 16 | 3 (19%) |
| 2 | 2198.3 - 4642.3 | 32 | 30 | 17 | 9 (53%) |

Two periods were detected at the expected boundary (period 2 starts at video 36.6 min, the
expected half-time mark); interval counts (36 in period 1) and canonical mirroring (player x
mostly under 47 near setup in both periods, ~90-97% of the sampled positions) match Phase 0. The
dead-ball setup rate is below Phase 0's 65% in both periods (worst in period 1), outside the
+/-10 point tolerance. Diagnosis: re-running the shared `halfcourt.build_records`/`find_setup` on
the original Phase 0 trajectories and scoreboard (`data/trajectories.jsonl`,
`data/scoreboard_raw.jsonl`) exactly reproduces the Phase 0 numbers (36 intervals, 34 located, 20
dead-ball, 13 with setup), which rules out an algorithm regression, a period-split issue, and an
attack-direction mis-vote (both periods vote unanimously for one basket, not tied). Period 1 has
the same 13 full-court intervals as Phase 0, but only 2 of 13 find a still, in-frontcourt frame in
the fresh full-game pass (Phase 0 found more), even widening the search window from 6 s to 20 s;
per-frame inspection shows too few Duke players simultaneously tracked and stationary in the
seconds after these full-court inbounds, while the aggregate per-period stillness rate is actually
slightly higher than Phase 0's (8.3% vs 4.3% of sampled frames). The likely cause is drift in the
hosted Roboflow Universe models (this run pulled a fresh `inference-gpu` build, months after Phase
0) or chunk-boundary effects from processing the full game in 16 parallel 5-minute chunks instead
of one dedicated clip, rather than a bug in possession segmentation, clock mapping, or outcomes.
This is a data-quality watchpoint for Phase 1B, not a code change in this task.

Only the `espn` layout has been exercised through a real ingest so far; `cbs`, `ncaa`, `cw` and
`cbssn` have fixtures and pass `probe_scoreboard.py` but no game using them has been run through
`ingest_game.py` yet.

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

After `annotate_outcomes.py` each possession also carries:

```json
{"clock_start": 1194.0, "clock_end": 1169.0,
 "score_before": {"Michigan": 0, "Duke": 0}, "score_after": {"Michigan": 2, "Duke": 0},
 "points_scored": 2, "scoreboard_points": 2, "outcome": "made_2",
 "events": [{"clock_text": "19:32", "team": "Michigan", "type": "DunkShot",
             "text": "Aday Mara makes 3-foot alley oop dunk", "scoring": true, "score_value": 2, ...}]}
```

- `clock_start` and `clock_end` are game-clock seconds remaining, read from the broadcast
  scoreboard with tesseract once per second; `score_before` and `score_after` come from the same
  OCR, validated against the score sequence in ESPN's play-by-play.
- `events` are the ESPN plays whose clock falls inside the possession, each with `t` (video
  seconds, from the scoreboard read that showed the event's clock) and `player` (roster name in
  the text), each assigned to exactly one possession (a shot on a shared boundary goes to the earlier possession, a foul or free throw
  to the later one). `outcome` is derived from the offense's events: `made_2`, `made_3`,
  `missed_2`, `missed_3`, `free_throws`, `turnover`, `foul`, or `null` when no event fell inside
  the window (dead-ball stretches). `points_scored` sums ESPN scoring plays; `scoreboard_points`
  is the OCR score delta for the same team, kept as a cross-check.

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
| Clock window resolved | 80 of 80 possessions |
| Outcomes | 19 made 2, 7 made 3, 7 missed 2, 16 missed 3, 11 turnovers, 9 free-throw trips, 2 fouls, 9 unlabelled |
| ESPN points vs scoreboard delta | 73 agree, 3 disagree (scoreboard graphic lags the play) |

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
- Possession outcomes depend on the OCR'd clock: with a 1 s clock resolution, several dead-ball
  segments at the same clock (a free-throw sequence) cannot be told apart, so the free throws land
  on one of them. A vision-only made/miss detector is described in `docs/backlog/vision-outcomes.md`.
- Jersey OCR at 720p misreads similar digits (2 vs 21, 3 vs 23); a number needs two agreeing
  reads on the team's roster, and concurrent duplicates on one team keep only the stronger vote.

## Tests

```bash
uv run pytest
```
