"""Cross-game coverage: acceptance per game, a markdown table, and the aggregated records file.

    uv run python scripts/coverage_report.py                    # data/plays/coverage.md + halfcourt.jsonl
    uv run python scripts/coverage_report.py --no-aggregate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays.games import GamePaths, load_registry

LOCATED_MIN = 0.85
AGREEMENT_MIN = 0.85
CLOCK_TRUST_MIN = 0.7


def acceptance(cov: dict) -> dict:
    periods = cov.get("periods", [])
    intervals = sum(p["intervals"] for p in periods)
    located = sum(p["located"] for p in periods)
    agree, disagree = cov.get("espn_agree", 0), cov.get("espn_disagree", 0)
    located_rate = located / intervals if intervals else 0.0
    agreement_rate = agree / (agree + disagree) if agree + disagree else 0.0
    clock_trust = cov.get("clock_trust_rate", 0.0)
    out = {
        "periods_ok": cov.get("detected_periods", len(periods)) == cov.get("expected_periods"),
        "located_ok": located_rate >= LOCATED_MIN,
        "agreement_ok": agreement_rate >= AGREEMENT_MIN,
        "clock_ok": clock_trust >= CLOCK_TRUST_MIN,
        "located_rate": located_rate, "agreement_rate": agreement_rate, "clock_trust": clock_trust,
    }
    out["ok"] = all(out[k] for k in ("periods_ok", "located_ok", "agreement_ok", "clock_ok"))
    return out


def game_row(game, root: Path) -> dict:
    paths = GamePaths.for_game(game.espn_id, root)
    row = {"game": game.espn_id, "opponent": game.opponent, "date": game.date, "layout": game.layout,
           "status": "pending"}
    if not paths.coverage.exists():
        return row
    cov = json.loads(paths.coverage.read_text())
    acc = acceptance(cov)
    dead = sum(p["dead_ball"] for p in cov["periods"])
    setups = sum(p["dead_ball_with_setup"] for p in cov["periods"])
    flags = [k for k in ("periods_ok", "located_ok", "agreement_ok", "clock_ok") if not acc[k]]
    row.update({"status": "ok" if acc["ok"] else "check", "periods": len(cov["periods"]),
                "located_rate": acc["located_rate"], "agreement_rate": acc["agreement_rate"],
                "dead_ball": dead, "setups": setups, "cost": cov.get("cost_estimate"), "flags": flags})
    return row


def render_table(rows: list[dict]) -> str:
    head = ("| Opponent | Date | Layout | Status | Periods | Located | ESPN agree | Dead-ball | Setups |"
            " Cost (est.) |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in rows:
        if r["status"] == "pending":
            lines.append(f"| {r['opponent']} | {r['date']} | {r['layout']} | pending | | | | | | |")
            continue
        flags = f" ({', '.join(r['flags'])})" if r.get("flags") else ""
        cost = f"${r['cost']:.2f}" if r.get("cost") is not None else ""
        lines.append(f"| {r['opponent']} | {r['date']} | {r['layout']} | {r['status']}{flags} | "
                     f"{r['periods']} | {r['located_rate']:.0%} | {r['agreement_rate']:.0%} | "
                     f"{r['dead_ball']} | {r['setups']} | {cost} |")
    return "\n".join(lines) + "\n"


def aggregate(rows: list[dict], root: Path, out: Path) -> int:
    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        for r in rows:
            if r["status"] == "pending":
                continue
            with open(GamePaths.for_game(r["game"], root).halfcourt) as f:
                for line in f:
                    w.write(line)
                    n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/games")
    ap.add_argument("--out", default="data/plays/coverage.md")
    ap.add_argument("--aggregate", default="data/plays/halfcourt.jsonl")
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    rows = [game_row(g, root) for g in sorted(load_registry(), key=lambda g: g.date)]
    done = [r for r in rows if r["status"] != "pending"]
    total_cost = sum(r["cost"] or 0 for r in done)
    text = (f"# Coverage: {len(done)} of {len(rows)} games ingested, "
            f"total GPU estimate ${total_cost:.2f} (measured about $0.064 per video minute)\n\n"
            + render_table(rows))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    if not args.no_aggregate:
        n = aggregate(rows, root, Path(args.aggregate))
        print(f"{n} half-court records -> {args.aggregate}")


if __name__ == "__main__":
    main()
