"""Phase 0 spike: Duke half-court records, setup snapshots and go/no-go numbers for one game.

    uv run python scripts/spike_setups.py data/trajectories.jsonl \
        --espn-cache data/espn_summary.json --raw-ocr data/scoreboard_raw.jsonl \
        --out data/plays/spike
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import halfcourt as H
from basketball_plays import montage as Mo
from basketball_plays import playbyplay as pbp
from basketball_plays import scoreboard as sb
from basketball_plays import zones as Z
from basketball_plays.schema import read_possessions

SENSITIVITY_MOVE_FT = (1.0, 1.5, 2.0)
SENSITIVITY_MIN_PLAYERS = (4, 3)


def pct(n: int, total: int) -> str:
    """`n/N (p%)`, the one way this report states a proportion."""
    return f"{n}/{total} ({n / total:.0%})" if total else f"{n}/0 (n/a)"


def record_tracks(r: H.HalfcourtRecord) -> list[H.Track]:
    """The record's stored (already canonical, already clipped) player tracks."""
    return [
        H.Track(p["track_id"], p["name"], p["jersey"],
                {round(row[0], 3): (row[1], row[2]) for row in p["trajectory"]}, set())
        for p in r.players
    ]


def sensitivity_table(halfcourt: list[H.HalfcourtRecord]) -> list[str]:
    """Setup rate over the half-court records for a grid of stillness thresholds.

    Re-runs `find_setup` itself, so the dead-ball window, the full-court exception, the
    frontcourt requirement and the tolerant lookback are all exactly as in the pipeline.
    """
    header = ["| max_move_ft | " + " | ".join(f"min players {n}" for n in SENSITIVITY_MIN_PLAYERS)
              + " |", "|---|" + "---|" * len(SENSITIVITY_MIN_PLAYERS)]
    rows = []
    for move in SENSITIVITY_MOVE_FT:
        cells = []
        for min_players in SENSITIVITY_MIN_PLAYERS:
            hits = 0
            for r in halfcourt:
                tracks = record_tracks(r)
                handler = {round(t, 3): (x, y) for t, x, y in r.ball_handler}
                _, no_setup = H.find_setup(
                    tracks, handler, r.start_type, r.t_start, r.t0, r.t_end,
                    full_court=r.full_court, max_move_ft=move, min_players=min_players,
                )
                hits += not no_setup
            cells.append(pct(hits, len(halfcourt)))
        rows.append(f"| {move} | " + " | ".join(cells) + " |")
    return header + rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trajectories")
    ap.add_argument("--game-id", default=pbp.GAME_ID)
    ap.add_argument("--team", default="Duke")
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--espn-cache", default="data/espn_summary.json")
    ap.add_argument("--raw-ocr", default="data/scoreboard_raw.jsonl")
    ap.add_argument("--out", default="data/plays/spike")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--start-types", default="ato,dead,live",
                    help="comma-separated start types to keep (default all)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="append a setup-rate table over stillness thresholds")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    possessions = read_possessions(args.trajectories)
    summary = pbp.fetch_summary(args.game_id, cache=Path(args.espn_cache))
    events = pbp.parse_events(summary["plays"], period=args.period)
    reads = sb.clean_timeline(sb.load_timeline(args.raw_ocr), valid_states=pbp.score_states(events))

    records = H.build_records(args.game_id, args.team, events, reads, possessions)
    with open(out / "halfcourt.jsonl", "w") as f:
        f.writelines(r.to_json() + "\n" for r in records)

    located = [r for r in records if r.located and r.t0 is not None]
    keep = set(args.start_types.split(","))
    halfcourt = [r for r in located if not r.transition and r.start_type in keep]
    with_setup = [r for r in halfcourt if not r.no_setup]
    order = {"ato": 0, "dead": 1, "live": 2, "period": 3}
    tiles = [
        Mo.render_setup_tile(r)
        for r in sorted(halfcourt, key=lambda r: (order.get(r.start_type, 9), -r.clock_start))
    ]
    cv2.imwrite(str(out / "setups.png"), Mo.grid(tiles, cols=args.cols))

    occ = np.zeros(Z.N_ZONES)
    for r in with_setup:
        occ += Z.occupancy(np.array([[x, y] for _, x, y in Mo.labelled_positions_at(r, r.setup)]))
    top_zones = sorted(zip(Z.ZONE_NAMES, occ), key=lambda p: -p[1])[:10]
    visible = Counter(r.n_visible_at_setup for r in with_setup)
    setup_rate = len(with_setup) / len(halfcourt) if halfcourt else 0.0
    suspect = [r for r in with_setup if r.suspect_duplicates]

    lines = [
        f"# Phase 0 spike: {args.team}, game {args.game_id}, period {args.period}", "",
        "| Metric | Value |", "|---|---|",
        f"| ESPN intervals for {args.team} | {len(records)} |",
        f"| Located in video (clock mapped, t0 found) | {pct(len(located), len(records))} |",
        f"| Excluded (transition or start type not kept) | {pct(len(located) - len(halfcourt), len(located))} |",
        f"| Half-court records | {pct(len(halfcourt), len(located))} |",
        f"| Full-court starts | {pct(sum(1 for r in halfcourt if r.full_court), len(halfcourt))} |",
        f"| Start types | {dict(Counter(r.start_type for r in halfcourt))} |",
        f"| With a setup frame | {pct(len(with_setup), len(halfcourt))} |",
        "| Setup rate by start type | "
        + ", ".join(
            f"{k}: " + pct(sum(1 for r in with_setup if r.start_type == k),
                           sum(1 for r in halfcourt if r.start_type == k))
            for k in sorted({r.start_type for r in halfcourt})
        )
        + " |",
        f"| Players visible at setup | {dict(sorted(visible.items()))} |",
        "| Records with more than 5 visible at setup | "
        + pct(len(suspect), len(with_setup)) + " |",
        f"| Outcomes | {dict(Counter(r.outcome or 'none' for r in halfcourt))} |", "",
        "Go criterion: setup rate >= 60% -> "
        + ("**MET**" if setup_rate >= 0.6 else "**NOT MET**"), "",
        "Most occupied zones at setup (player-frames):", "",
        *[f"- {name}: {int(n)}" for name, n in top_zones if n > 0], "",
        (
            f"Montage: `{out / 'setups.png'}` (tiles sorted ato, dead, live; header shows index, "
            "start type, clock, outcome, players visible; `T` = transition, `?` = no setup)."
        ),
        "Unlocated intervals: "
        + ", ".join(
            f"#{r.index} {r.start_type} {r.clock_start:.0f}" for r in records if r not in located
        ),
    ]
    if args.sensitivity:
        lines += ["", "## Threshold sensitivity", "",
                  f"Setup rate over the {len(halfcourt)} half-court records.", "",
                  *sensitivity_table(halfcourt)]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
