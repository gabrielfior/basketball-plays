"""Write a CSV summary of trajectories.jsonl for reviewing cases.

    uv run python scripts/summarize_possessions.py data/trajectories.jsonl --out data/possessions.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays.render import clip_name, clock_text
from basketball_plays.schema import read_possessions


def median_speed(p) -> float:
    """Median frame-to-frame speed over all tracks (ft/s); real players stay well under 15."""
    speeds = []
    for q in p.players:
        tr = np.array([r for r in q.trajectory if np.all(np.isfinite(r))])
        if len(tr) > 1:
            dt = np.diff(tr[:, 0])
            d = np.linalg.norm(np.diff(tr[:, 1:], axis=0), axis=1)
            speeds.extend((d[dt > 0] / dt[dt > 0]).tolist())
    return round(float(np.median(speeds)), 1) if speeds else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trajectories")
    ap.add_argument("--out", default="data/possessions.csv")
    args = ap.parse_args()
    rows = []
    for p in read_possessions(args.trajectories):
        named = sorted({q.name for q in p.players if q.name})
        shown = [e for e in (p.events or []) if "subbing" not in e.get("text", "")]
        rows.append({
            "possession_id": p.possession_id,
            "video_start_s": p.start_time, "video_end_s": p.end_time,
            "video_start": f"{int(p.start_time // 60)}:{int(p.start_time % 60):02d}",
            "clock_start": clock_text(p.clock_start) if p.clock_start is not None else "",
            "clock_end": clock_text(p.clock_end) if p.clock_end is not None else "",
            "offense": p.offense_team or "", "basket": p.attacking_basket,
            "outcome": p.outcome or "", "points": p.points_scored if p.points_scored is not None else "",
            "scoreboard_points": p.scoreboard_points if p.scoreboard_points is not None else "",
            "score_before": f"{p.score_before['Michigan']}-{p.score_before['Duke']}" if p.score_before else "",
            "score_after": f"{p.score_after['Michigan']}-{p.score_after['Duke']}" if p.score_after else "",
            "n_tracks": len(p.players), "n_named": len(named), "named_players": "; ".join(named),
            "median_speed_ft_s": median_speed(p), "erratic": "yes" if median_speed(p) > 15 else "",
            "events": " | ".join(f"{e.get('clock_text', '')} {e.get('text', '')}" for e in shown),
            "clip": clip_name(p),
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} possessions")


if __name__ == "__main__":
    main()
