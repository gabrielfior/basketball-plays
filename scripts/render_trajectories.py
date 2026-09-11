"""Render the first N possessions from trajectories.jsonl as a 2D court animation (MP4).

    uv run python scripts/render_trajectories.py data/trajectories.jsonl --n 5 --out data/trajectories.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from basketball_plays.court import NCAA, draw_court
from basketball_plays.render import VideoWriter, possession_times, render_court_frame
from basketball_plays.schema import read_possessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectories", help="trajectories.jsonl produced by extract_trajectories.py")
    ap.add_argument("--n", type=int, default=5, help="number of possessions to render")
    ap.add_argument("--out", default="data/trajectories.mp4")
    ap.add_argument("--scale", type=float, default=10.0, help="pixels per foot")
    ap.add_argument("--trail", type=float, default=2.0, help="trail length in seconds")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()

    possessions = read_possessions(args.trajectories)[: args.n]
    if not possessions:
        sys.exit("no possessions found")
    court = draw_court(NCAA, args.scale, 30)
    first = render_court_frame(possessions[0], possessions[0].start_time, args.scale, 30, args.trail,
                               court_img=court)
    writer = VideoWriter(args.out, possessions[0].fps * args.speed, (first.shape[1], first.shape[0]))
    for pos in tqdm(possessions, desc="possessions"):
        for t in possession_times(pos):
            writer.write(render_court_frame(pos, t, args.scale, 30, args.trail, court_img=court))
    writer.close()
    print(f"wrote {args.out}: {writer.n} frames, {len(possessions)} possessions")


if __name__ == "__main__":
    main()
