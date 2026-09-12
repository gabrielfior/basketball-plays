"""Draw a layout's scoreboard regions on a frame and print what tesseract reads from each.

    uv run python scripts/probe_scoreboard.py tests/fixtures/scoreboards/cbs_fsu.jpg --layout cbs
    uv run python scripts/probe_scoreboard.py data/games/401820644/video.mp4 --t 900 --layout cbs
    uv run python scripts/probe_scoreboard.py frame.jpg \
        --regions "away=100,620,190,660;clock=1030,620,1110,660;home=560,620,650,660"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import broadcasts as B
from basketball_plays import scoreboard as sb


def load_frame(path: str, t: float):
    if path.lower().endswith((".jpg", ".png")):
        return cv2.imread(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def parse_regions(text: str) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for part in text.split(";"):
        key, vals = part.split("=")
        out[key.strip()] = tuple(int(v) for v in vals.split(","))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="image or video")
    ap.add_argument("--t", type=float, default=0.0, help="video time in seconds")
    ap.add_argument("--layout", default="espn")
    ap.add_argument("--regions", default=None, help="override: key=x1,y1,x2,y2;...")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--out", default="probe.jpg")
    args = ap.parse_args()
    frame = load_frame(args.source, args.t)
    if frame is None:
        sys.exit("could not read a frame")
    frame = cv2.resize(frame, (1280, 720))
    if args.regions:  # an override measures a layout that is not registered yet
        regions, invert = parse_regions(args.regions), args.invert
    else:
        lay = B.get_layout(args.layout)
        regions, invert = lay.regions, args.invert or lay.invert
    read = sb.read_frame(frame, args.t, regions=regions, invert=invert)
    print(f"clock={read.clock_text!r} ({read.clock}) away={read.away} home={read.home}")
    out_dir = Path(args.out).resolve().parent
    for key, (x1, y1, x2, y2) in regions.items():
        crop = frame[y1:y2, x1:x2]
        if invert:
            crop = cv2.bitwise_not(crop)
        cv2.imwrite(str(out_dir / f"probe_{key}.jpg"), sb.preprocess(crop))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, key, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(args.out, frame)
    print(f"wrote {args.out} and probe_<key>.jpg crops")


if __name__ == "__main__":
    main()
