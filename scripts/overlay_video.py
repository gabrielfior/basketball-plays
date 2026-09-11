"""Overlay detected player boxes and identities on the source video for the first N possessions.

    uv run python scripts/overlay_video.py data/duke_michigan_q1.mp4 data/trajectories.jsonl --n 5 --out data/overlay.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays.court import NCAA, draw_court  # noqa: E402
from basketball_plays.render import VideoWriter, box_lookup, draw_overlay, render_court_frame  # noqa: E402
from basketball_plays.schema import read_possessions  # noqa: E402
from basketball_plays.video import probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("trajectories")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="data/overlay.mp4")
    ap.add_argument("--no-inset", action="store_true", help="skip the court inset")
    args = ap.parse_args()

    possessions = read_possessions(args.trajectories)[: args.n]
    if not possessions:
        sys.exit("no possessions found")
    meta = probe(args.video)
    writer = VideoWriter(args.out, meta.fps, (meta.width, meta.height))
    court = draw_court(NCAA, 4.0, 10)
    cap = cv2.VideoCapture(args.video)
    try:
        for pos in tqdm(possessions, desc="possessions"):
            boxes = box_lookup(pos)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(pos.start_time * meta.fps)))
            n_frames = int(round((pos.end_time - pos.start_time) * meta.fps))
            for k in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                t_src = pos.start_time + k / meta.fps
                t = round(pos.start_time + round((t_src - pos.start_time) * pos.fps) / pos.fps, 3)
                frame = draw_overlay(frame, pos, t, boxes)
                if not args.no_inset:
                    inset = render_court_frame(pos, t, 4.0, 10, 2.0, header=24, court_img=court)
                    h, w = inset.shape[:2]
                    y0, x0 = meta.height - h - 10, meta.width - w - 10
                    frame[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(frame[y0:y0 + h, x0:x0 + w], 0.15, inset, 0.85, 0)
                writer.write(frame)
    finally:
        cap.release()
        writer.close()
    print(f"wrote {args.out}: {writer.n} frames, {len(possessions)} possessions")


if __name__ == "__main__":
    main()
