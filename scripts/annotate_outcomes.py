"""Attach game clock, scores and play-by-play outcomes to each possession in trajectories.jsonl.

Sources: the broadcast scoreboard (tesseract OCR, once per second) for the game clock and the
score, and ESPN's play-by-play for event labels. The clock links the two: a possession's video
time range becomes a game-clock range, and the ESPN events inside that range describe it.

    uv run python scripts/annotate_outcomes.py data/duke_michigan_q1.mp4 data/trajectories.jsonl
    uv run python scripts/annotate_outcomes.py data/duke_michigan_q1.mp4 data/trajectories.jsonl --skip-ocr
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import gameinfo
from basketball_plays import playbyplay as pbp
from basketball_plays import scoreboard as sb
from basketball_plays.schema import read_possessions, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("trajectories")
    ap.add_argument("--out", default=None, help="output path (default: overwrite the input)")
    ap.add_argument("--raw-ocr", default="data/scoreboard_raw.jsonl")
    ap.add_argument("--espn-cache", default="data/espn_summary.json")
    ap.add_argument("--skip-ocr", action="store_true", help="reuse --raw-ocr instead of reading the video")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between scoreboard reads")
    ap.add_argument("--clock-margin", type=float, default=1.0, help="seconds of slack on the clock window")
    ap.add_argument("--max-dt", type=float, default=15.0, help="how far to look for a trusted clock read")
    ap.add_argument("--period", type=int, default=1, help="ESPN period number to annotate against")
    ap.add_argument("--t-lo", type=float, default=None,
                    help="restrict to possessions/reads overlapping [t-lo, t-hi] (video seconds)")
    ap.add_argument("--t-hi", type=float, default=None,
                    help="restrict to possessions/reads overlapping [t-lo, t-hi] (video seconds)")
    args = ap.parse_args()

    possessions = read_possessions(args.trajectories)
    start, end = possessions[0].start_time, possessions[-1].end_time

    if args.skip_ocr and Path(args.raw_ocr).exists():
        raw = sb.load_timeline(args.raw_ocr)
    else:
        print(f"reading the scoreboard every {args.every}s from {start:.0f}s to {end:.0f}s ...")
        raw = sb.read_timeline(args.video, start, end + 1, every_s=args.every)
        sb.write_timeline(args.raw_ocr, raw)

    summary = pbp.fetch_summary(cache=Path(args.espn_cache))
    info = gameinfo.from_summary(summary)
    events = pbp.parse_events(summary["plays"], period=args.period, team_by_id=info.team_by_id)

    span = args.t_lo is not None and args.t_hi is not None
    raw_for_clean = [r for r in raw if args.t_lo <= r.t <= args.t_hi] if span else raw
    targets = possessions
    if span:
        targets = [p for p in possessions if p.start_time <= args.t_hi and p.end_time >= args.t_lo]

    reads = sb.clean_timeline(raw_for_clean, valid_states=pbp.score_states(events))
    print(f"scoreboard: {len(reads)} reads, clock trusted on {sum(r.clock is not None for r in reads) / len(reads):.0%}, "
          f"score on {sum(r.away is not None for r in reads) / len(reads):.0%}; {len(events)} play-by-play events")

    for pos in targets:
        pos.clock_start = sb.value_at(reads, pos.start_time, "clock", args.max_dt)
        pos.clock_end = sb.value_at(reads, pos.end_time, "clock", args.max_dt)
    assigned = pbp.assign_events(
        [(p.possession_id, p.clock_start, p.clock_end, p.offense_team) for p in targets], events,
        margin=args.clock_margin,
    )

    stats = Counter()
    for pos in targets:
        c0, c1 = pos.clock_start, pos.clock_end
        a0, h0 = sb.value_at(reads, pos.start_time, "away", args.max_dt), sb.value_at(reads, pos.start_time, "home", args.max_dt)
        a1, h1 = sb.value_at(reads, pos.end_time + 3, "away", args.max_dt), sb.value_at(reads, pos.end_time + 3, "home", args.max_dt)
        pos.score_before = {info.away: a0, info.home: h0} if a0 is not None and h0 is not None else None
        pos.score_after = {info.away: a1, info.home: h1} if a1 is not None and h1 is not None else None
        if pos.score_before and pos.score_after and pos.offense_team in pos.score_before:
            pos.scoreboard_points = pos.score_after[pos.offense_team] - pos.score_before[pos.offense_team]
        if c0 is not None and c1 is not None:
            window = assigned.get(pos.possession_id, [])
            times = pbp.locate_events(window, reads, pos.start_time, pos.end_time, c0, c1)
            pos.events = []
            for e, t in zip(window, times):
                d = e.to_dict()
                d["t"] = t
                d["player_team"], d["player"] = pbp.match_player(e.text, info.rosters)
                pos.events.append(d)
            pos.outcome, pos.points_scored = pbp.derive_outcome(window, pos.offense_team)
            stats["with_clock"] += 1
        stats[pos.outcome or "unlabelled"] += 1
        if pos.points_scored is not None and pos.scoreboard_points is not None:
            stats["agree" if pos.points_scored == pos.scoreboard_points else "disagree"] += 1

    out = args.out or args.trajectories
    write_jsonl(out, possessions)
    n = len(targets)
    print(f"{n} possessions annotated -> {out}")
    print(f"  clock window resolved: {stats['with_clock']}/{n}")
    print("  outcomes: " + ", ".join(f"{k}={v}" for k, v in stats.most_common() if k not in ("with_clock", "agree", "disagree")))
    print(f"  ESPN points vs scoreboard delta: agree={stats['agree']} disagree={stats['disagree']}")
    for pos in targets[:12]:
        c = f"{pos.clock_start:.0f}->{pos.clock_end:.0f}" if pos.clock_start is not None and pos.clock_end is not None else "?"
        print(f"  #{pos.possession_id:3d} {pos.start_time:7.1f}-{pos.end_time:7.1f}s clock {c:>10s} {pos.offense_team or '?':9s} "
              f"{pos.outcome or '-':12s} pts={pos.points_scored} sb={pos.scoreboard_points} events={len(pos.events or [])}")


if __name__ == "__main__":
    main()
