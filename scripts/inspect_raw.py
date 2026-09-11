"""Quick statistics over data/raw/frames_*.jsonl to sanity-check the GPU stage.

    uv run python scripts/inspect_raw.py data/raw
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays.extract import project_frame  # noqa: E402
from basketball_plays.schema import PLAYER_CLASSES, read_frames  # noqa: E402


def main() -> None:
    raw_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    frames = []
    for p in sorted(raw_dir.glob("frames_*.jsonl")):
        frames.extend(read_frames(p))
    frames.sort(key=lambda f: f.t)
    if not frames:
        sys.exit("no frames")
    n = len(frames)
    n_kp = np.array([sum(1 for k in f.keypoints if k[2] > 0.5) for f in frames])
    n_players = np.array([sum(1 for d in f.detections if d.cls in PLAYER_CLASSES) for f in frames])
    has_ball = np.array([f.ball is not None for f in frames])
    shots = sum(f.shot for f in frames)
    projected = [project_frame(f) for f in frames]
    has_h = np.array([p.fit is not None for p in projected])
    on_court = np.array([len(p.players) for p in projected])
    reads = Counter(str(x["text"]) for f in frames for x in f.numbers)
    clusters = Counter(d.team_cluster for f in frames for d in f.detections)
    print(f"frames: {n}  span {frames[0].t:.1f}s .. {frames[-1].t:.1f}s  shot changes: {shots}")
    print(f"keypoints>0.5 per frame: mean {n_kp.mean():.1f}  frames with >=4: {(n_kp >= 4).mean():.1%}")
    print(f"homography ok: {has_h.mean():.1%}   players/frame: mean {n_players.mean():.1f}  "
          f"on-court after projection: mean {on_court.mean():.1f}  frames with >=6 on court: {(on_court >= 6).mean():.1%}")
    print(f"ball detected: {has_ball.mean():.1%}   team clusters: {dict(clusters)}")
    print(f"OCR reads: {sum(reads.values())} total, top: {reads.most_common(15)}")
    valid = has_h & (on_court >= 6)
    runs, cur = [], 0
    for v in valid:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if runs:
        runs = np.array(runs) / 10.0
        print(f"valid runs: {len(runs)}  median {np.median(runs):.1f}s  max {runs.max():.1f}s  "
              f">=3s: {(runs >= 3).sum()}")


if __name__ == "__main__":
    main()
