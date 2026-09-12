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
  coverage.json        per-period counts plus whole-game OCR/ESPN quality (see below)
  spans_used.json      the period spans extraction actually used, to detect drift later
  cost.json            the gpu step's cost estimate, parsed from its output
  game.json            the registry entry, written after a full run
```

`scripts/ingest_game.py <espn_id>` runs the seven steps (`download`, `espn`, `gpu`, `ocr`,
`extract`, `annotate`, `halfcourt`) in order, skipping any step whose output already exists so a
failed run can be fixed and the same command re-run to resume. OCR runs before extraction so the
scoreboard's period spans are already known when Stage B decides offense: each period has one
fixed basket assignment (teams switch baskets at half time), so offense is learned per period
from those spans rather than guessed from a single whole-video vote.

```bash
uv run python scripts/ingest_game.py 401817238 --dry-run     # print the plan, run nothing
uv run python scripts/ingest_game.py 401817238               # run every step, skip existing outputs
uv run python scripts/ingest_game.py 401817238 --steps gpu    # run one step
uv run python scripts/ingest_game.py 401817238 --force halfcourt --steps halfcourt  # redo a step
uv run python scripts/ingest_game.py 401817238 --max-cost 5   # lower the GPU cost ceiling (10)
uv run python scripts/ingest_game.py 401817238 --allow-period-mismatch  # ignore a clash
```

`--force` is transitive: `--force ocr` also redoes `extract`, `annotate` and `halfcourt`, because
a redone step invalidates everything computed from it. The expanded set is printed before the run
starts.

Before extraction, annotation and the half-court records, the number of period spans detected in
the scoreboard timeline is cross-checked against the highest period number in ESPN's play-by-play
(the count that says whether the game went to overtime). A disagreement means the spans are not
the game's periods, so the run stops with exit code 2 and prints both counts and the detected
boundaries; `--allow-period-mismatch` continues anyway and records `"period_check": "mismatch"` in
`coverage.json`. A scoreboard timeline yielding *no* spans is likewise fatal in
`extract_trajectories.py` (pass `--allow-no-periods` to fall back to the time-windowed offense
rule instead). Extraction records the spans it used in `spans_used.json`; the `halfcourt` step
warns when the spans it detects no longer match them.

The `gpu` step prints an estimated cost (video minutes x $0.09, a conservative ceiling: the
Michigan game actually billed $0.064/min) before doing anything remote and refuses to proceed
above `--max-cost`; the estimate is saved to `cost.json` and reported as `cost_estimate` in
`coverage.json`. Chunks on Modal resume: a chunk whose `frames_XX.jsonl` is already on the volume
returns immediately, so a run that died part-way is retried with the same command plus
`--skip-upload --skip-fit` (printed as a hint when a run fails, and `--force-chunks` redoes the
finished chunks).

`coverage.json` carries, besides the per-period counts (`intervals`, `located`, `dead_ball`,
`dead_ball_with_setup`, `start_types`): `layout`, `expected_periods`, `detected_periods`,
`period_check`, `ocr_reads`, `espn_agree` / `espn_disagree` (possessions where ESPN's points and
the scoreboard delta both exist, and agree or not), and `cost_estimate` when the `gpu` step ran.
`clock_trust_rate` and `score_trust_rate` (the fraction of that period's scoreboard reads whose
clock, and whose score, survived cleaning) appear in every `periods[]` entry alongside that
period's `ocr_reads`, and at the top level as the read-count-weighted average over the periods.
They are computed per period on purpose: `clean_timeline` enforces a non-increasing clock, so
cleaning a whole game in one pass rejects nearly everything after the clock resets at half time.

When the automatic cluster-to-team decision is wrong for a game, re-run extraction with
`--team-map "0=<team>"` (e.g. `--team-map "0=North Carolina"`): the name must be one of that
game's two teams, and the other cluster gets the other team.

Each game's broadcast uses one of five scoreboard graphic layouts
(`basketball_plays/broadcasts.py`), set per game in `games.json`: `espn` (ESPN, ESPN2, ACC
Network), `cbs`, `ncaa` (NCAA tournament), `cw` (The CW) and `cbssn` (CBS Sports Network); each
has a fixture frame under `tests/fixtures/scoreboards/`. `ingest_game.py` passes the registry's
layout to both the `ocr` step and the `annotate` step (`--layout`, used only when annotation is
asked to read the video itself rather than reuse the OCR timeline), along with the game's ESPN id
(`--espn-id`); run standalone, both scripts default to the `espn` layout and the reference
game. Check a layout's regions against a real frame with:

```bash
uv run python scripts/probe_scoreboard.py data/games/<espn_id>/video.mp4 --t 900 --layout cbs
```

### Smoke test: the whole Michigan game (401817238)

`uv run python scripts/ingest_game.py 401817238` end to end on the full 77.7-minute broadcast
(Phase 0 only covered the first half): download (1.98 GB), ESPN summary, GPU stage (16 chunks,
46,618 frames at 10 fps; cost estimate `77.7 video minutes x $0.09 = $6.99`, actual Modal billing
**$5.00 total** — L4 GPU $4.28, CPU $0.59, memory $0.13), scoreboard OCR (4,634 reads, run before
extraction), extraction (157 possessions, offense learned per period from the OCR'd period
spans), per-period annotation and half-court records:

| Period | Span (video s) | Intervals | Located | Dead-ball | With setup |
|---|---|---|---|---|---|
| 1 | 10.3 - 2198.3 | 36 | 35 | 20 | 10 (50%) |
| 2 | 2198.3 - 4642.3 | 32 | 30 | 17 | 9 (53%) |

Two periods were detected at the expected boundary (period 2 starts at video 36.6 min, the
expected half-time mark), matching the two periods in ESPN's play-by-play (`"period_check":
"ok"`). The scoreboard timeline has 4,634 reads over the whole game; cleaned per period, the
clock is trusted on 85.2% of period 1's 2,198 reads and 86.9% of period 2's 2,446, and the score
on 81.4% and 86.8% (86.1% and 84.2% weighted over the game). ESPN's per-possession points agree
with the scoreboard delta on 144 possessions against 9 disagreements.

An earlier run of this smoke test showed a much lower period-1 setup rate (3 of 16 dead-ball
intervals, 19%) than period 2's. The cause was the offense map: it was learned once over the
*whole* video, but teams switch baskets at half time, so that single map's votes partly canceled
between the two halves and period 1's offense label was wrong for roughly half its possessions —
which made `find_setup`'s frontcourt/attacking-basket check miss most of period 1's real setups.
The fix (this task) learns one offense map per period instead, from the scoreboard's period
spans (`possessions.learn_offense_maps_by_span`); period 1's setup rate above (50%) is now in
line with period 2's (53%).

The remaining gap to Phase 0 on the same footage (10 vs. 13 setups found in period 1) is not the
offense-map bug: it comes from tracking differences between two separate GPU runs of the same
clip (this smoke test re-ran Stage A rather than reusing Phase 0's raw detections) — 7 of Phase
0's 13 setup records changed verdict, each with its `t0` within 1 s of Phase 0's `t0`, consistent
with slightly different player tracks or box positions at the same moment rather than a different
possession being picked. This is a Phase 2 tracking-robustness item (consistency across separate
GPU runs of the same footage), not a bug in possession segmentation, clock mapping, or the
offense/outcome logic.

Only the `espn` layout has been exercised through a real ingest so far; `cbs`, `ncaa`, `cw` and
`cbssn` have fixtures and pass `probe_scoreboard.py` but no game using them has been run through
`ingest_game.py` yet.

## Set discovery and labelling

```bash
uv run python scripts/refresh_plays.py                # full refresh: coverage, features, clusters, defence, page
uv run python scripts/refresh_plays.py --skip-cluster  # keep clusters.json and montages/ (cluster ids stay put)
```

Runs, in order: `coverage_report.py` (aggregates every game's `halfcourt.jsonl` into
`data/plays/halfcourt.jsonl` and writes `data/plays/coverage.md`), `build_features.py`
(per-record feature vectors), `cluster_plays.py` (candidate sets by clustering, skipped by
`--skip-cluster`), `classify_defense.py` (man/zone per possession) and `build_page.py` (the
labelling page). Each step prints its own summary; the refresh script prints one `==` line per
step plus the final page path and size.

Outputs, all under `data/plays/` (gitignored except `labels.json`):

| File | From | What it is |
|---|---|---|
| `halfcourt.jsonl` | `coverage_report.py` | every ingested game's half-court records, concatenated |
| `coverage.md` | `coverage_report.py` | per-game acceptance table |
| `features.npz` + `features_index.json` | `build_features.py` | per-record feature vectors (`X`, `H`, `S` arrays) plus row metadata; see the parquet note below |
| `clusters.json` | `cluster_plays.py` | cluster assignments, centroids, k, silhouette, stability, per-cluster stats |
| `montages/cluster_<c>.png` | `cluster_plays.py` | nine setup-frame snapshots nearest each cluster's centroid |
| `defense.json` | `classify_defense.py` | man/zone/unknown label plus confidence per possession |
| `page/index.html` | `build_page.py` | the offline labelling page (self-contained, works from `file://`) |
| `labels.json` | the page's Export tab | the user's cluster names, validation labels and defence corrections; committed |

Naming clusters and exporting labels: open `data/plays/page/index.html`, go to the Clusters tab,
name every cluster you recognise (merge look-alikes, discard noise) -- decisions autosave to this
browser's `localStorage` as you go. When ready to keep them, open the Export tab, copy the JSON
and paste it over `data/plays/labels.json`, then commit that file. Re-clustering after adding
games renumbers the clusters (KMeans is refit on a different, larger row set), so names keyed by
cluster id in `labels.json` can end up pointing at the wrong set: export first, then either accept
that the new clusters need renaming, or run the refresh with `--skip-cluster` to keep the old
`clusters.json` (and therefore the old ids) while you finish naming. A future task may add
centroid matching to carry names across a re-cluster automatically.

Numbers from a refresh on the 7 games ingested so far (out of 27 registered): 254 dead-ball
feature rows, clustering fit on 155 setup-frame rows (99 more assigned to the nearest centroid
afterwards), k=9, silhouette 0.088, stability ARI 0.454, defence labels man 78 / zone 58 / unknown
118, page 3.72 MB. The validation sample (36 possessions) is drawn entirely from held-out splits,
which today means the one `ncaa`-layout game ingested (Siena) -- every other ingested game is
still `train`, so this is a small and unbalanced corpus; treat cluster shapes, the defence rule
and validation-sample coverage as provisional until more games, especially more held-out ones,
are ingested.

`features.npz` / `features_index.json` replace the parquet file an earlier version of this
pipeline used: parquet needs pandas and pyarrow, and a plain NumPy `.npz` (arrays) plus a JSON
index (row metadata, roster order) covers the same data without adding that dependency.

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
