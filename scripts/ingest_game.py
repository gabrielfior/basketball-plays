"""Ingest one game end to end, resumably.

Steps (each skipped when its output exists):
  download   yt-dlp 720p avc1 video -> data/games/<id>/video.mp4
  espn       ESPN summary -> espn_summary.json
  gpu        Modal Stage A -> raw/frames_XX.jsonl (prints a cost estimate first, saved to cost.json)
  ocr        scoreboard timeline -> scoreboard_raw.jsonl (whole video; layout from games.json)
  extract    Stage B -> trajectories.jsonl (rosters from the ESPN summary; offense learned per
             period from the scoreboard timeline, not guessed from a time window)
  annotate   clock, score, outcome per possession, per period
  halfcourt  Duke half-court records per period -> halfcourt.jsonl, coverage.json

OCR now runs before extract (it no longer needs possession times to pick its range - it reads
the whole video) so extract can pass the scoreboard timeline's period spans straight in.

Every step that depends on the period spans first cross-checks them against the ESPN summary:
the number of spans `periods.period_spans` found must equal the highest ESPN period number, or
the run stops with exit code 2 (`--allow-period-mismatch` overrides). A span's index is only a
period number when that check passes, so annotation, `period_length` and the half-court records
cannot be silently attached to the wrong period.

`--force` is transitive: forcing a step also forces everything downstream of it, since a redone
step invalidates the outputs computed from it.

    uv run python scripts/ingest_game.py 401817238 --dry-run
    uv run python scripts/ingest_game.py 401817238
    uv run python scripts/ingest_game.py 401817238 --steps halfcourt --force halfcourt
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import gameinfo
from basketball_plays import halfcourt as H
from basketball_plays import periods as P
from basketball_plays import playbyplay as pbp
from basketball_plays import scoreboard as sb
from basketball_plays.games import GamePaths, get_game
from basketball_plays.schema import read_possessions, write_jsonl

STEPS = ["download", "espn", "gpu", "ocr", "extract", "annotate", "halfcourt"]
YTDLP_FORMAT = "bestvideo[height<=720][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]"

# A redone step invalidates everything computed from it, so --force is expanded transitively.
DOWNSTREAM = {
    # download is the source of every later step (espn included: a re-download means a new
    # video, and the whole game is being redone).
    "download": ["espn", "gpu", "ocr", "extract", "annotate", "halfcourt"],
    "espn": ["extract", "annotate", "halfcourt"],
    "gpu": ["extract", "annotate", "halfcourt"],
    "ocr": ["extract", "annotate", "halfcourt"],
    "extract": ["annotate", "halfcourt"],
    "annotate": ["halfcourt"],
    "halfcourt": [],
}
COST_LINE = re.compile(r"estimated cost:.*=\s*\$([0-9]+(?:\.[0-9]+)?)")


def expand_force(force: set[str]) -> set[str]:
    """Close `force` over DOWNSTREAM (forcing a step forces everything computed from it)."""
    out = set(force)
    queue = list(force)
    while queue:
        for nxt in DOWNSTREAM.get(queue.pop(), []):
            if nxt not in out:
                out.add(nxt)
                queue.append(nxt)
    return out


def run(cmd: list[str], dry: bool, capture: bool = False) -> str:
    """Run `cmd`, echoing it first. With `capture`, also tee its output and return it."""
    print("+", " ".join(cmd))
    if dry:
        return ""
    if not capture:
        subprocess.run(cmd, check=True)
        return ""
    lines: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return "".join(lines)


def video_duration(video: Path) -> float:
    """Duration in seconds via ffprobe; ffprobe reports `N/A` (or nothing) for a broken file."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True, check=True)
    text = probe.stdout.strip()
    try:
        return float(text)
    except ValueError:
        raise SystemExit(
            f"ffprobe could not read a duration from {video} (got {text!r}); "
            "the file is missing, truncated or not a video") from None


def spans_as_json(spans: list[P.PeriodSpan]) -> list[dict]:
    return [{"period": s.period, "t_lo": round(s.t_lo, 3), "t_hi": round(s.t_hi, 3)}
            for s in spans]


def check_periods(game, paths, allow_mismatch: bool):
    """Cross-check the OCR'd period spans against the ESPN summary's period count.

    Span index + 1 is only an ESPN period number when the counts agree, so every consumer of
    `span.period` (annotation, `pbp.period_length`, the half-court records) must go through
    here first. Returns `(summary, info, spans, coverage_fields)`.
    """
    summary = json.loads(paths.espn_summary.read_text())
    info = gameinfo.from_summary(summary)
    spans = P.period_spans(sb.load_timeline(paths.scoreboard_raw))
    expected = max(info.periods) if info.periods else 0
    ok = len(spans) == expected
    fields = {"expected_periods": expected, "detected_periods": len(spans),
              "period_check": "ok" if ok else "mismatch"}
    if not ok:
        bounds = ", ".join(f"{s.t_lo:.1f}-{s.t_hi:.1f}s" for s in spans) or "(none)"
        print(f"period mismatch for game {game.espn_id} ({game.opponent}, {game.date}): "
              f"the scoreboard timeline {paths.scoreboard_raw} yields {len(spans)} period(s), "
              f"ESPN's play-by-play has {expected}")
        print(f"  detected span boundaries: {bounds}")
        if not allow_mismatch:
            print("  re-run the ocr step (a bad layout or a noisy timeline is the usual cause), "
                  "or pass --allow-period-mismatch to continue with the detected spans")
            raise SystemExit(2)
        print("  continuing anyway (--allow-period-mismatch)")
    return summary, info, spans, fields


def step_download(game, paths, dry):
    paths.root.mkdir(parents=True, exist_ok=True)
    run(["uvx", "yt-dlp", "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
         "-o", str(paths.video), game.youtube_url], dry)


def step_gpu(game, paths, dry, max_cost):
    out = run(["uv", "run", "modal", "run", "modal_app.py", "--video", str(paths.video),
               "--game", game.espn_id, "--end", "-1", "--out-dir", str(paths.raw_dir),
               "--max-cost", str(max_cost)], dry, capture=True)
    m = COST_LINE.search(out)
    if m and not dry:
        paths.root.mkdir(parents=True, exist_ok=True)
        (paths.root / "cost.json").write_text(json.dumps({"estimate_usd": float(m.group(1))},
                                                         indent=2))


def step_espn(game, paths, dry):
    if dry:
        print(f"+ fetch ESPN summary {game.espn_id} -> {paths.espn_summary}")
        return
    pbp.fetch_summary(game.espn_id, cache=paths.espn_summary)


def step_extract(game, paths, dry, allow_mismatch=False):
    if not paths.scoreboard_raw.exists() and not dry:
        # extract needs the OCR'd period spans to learn one offense map per period; run OCR
        # first if it hasn't happened yet (mirrors the espn auto-run below).
        step_ocr(game, paths, False)
    if not dry:
        _, _, spans, _ = check_periods(game, paths, allow_mismatch)
        (paths.root / "spans_used.json").write_text(json.dumps(spans_as_json(spans), indent=2))
    run(["uv", "run", "python", "scripts/extract_trajectories.py", str(paths.video), "--skip-gpu",
         "--raw-dir", str(paths.raw_dir), "--out", str(paths.trajectories),
         "--rosters-from", str(paths.espn_summary),
         "--periods-from", str(paths.scoreboard_raw)], dry)


def step_ocr(game, paths, dry):
    if dry:
        print(f"+ scoreboard OCR layout={game.layout} -> {paths.scoreboard_raw}")
        return
    duration = video_duration(paths.video)
    raw = sb.read_timeline(str(paths.video), 0.0, duration + 1.0, every_s=1.0, layout=game.layout)
    sb.write_timeline(paths.scoreboard_raw, raw)


def step_annotate(game, paths, dry, allow_mismatch=False):
    if dry:
        print(f"+ annotate per period -> {paths.trajectories}")
        return
    _, info, spans, _ = check_periods(game, paths, allow_mismatch)
    for span in spans:
        run(["uv", "run", "python", "scripts/annotate_outcomes.py",
             str(paths.video), str(paths.trajectories), "--skip-ocr",
             "--raw-ocr", str(paths.scoreboard_raw), "--espn-cache", str(paths.espn_summary),
             "--espn-id", game.espn_id, "--layout", game.layout,
             "--period", str(span.period), "--t-lo", str(span.t_lo), "--t-hi", str(span.t_hi)], dry)
    print(f"annotated {len(spans)} periods for {info.home} vs {info.away}")


def warn_if_spans_changed(paths, spans) -> None:
    """Warn when the spans now detected differ from the ones extraction actually used."""
    used_path = paths.root / "spans_used.json"
    if not used_path.exists():
        return
    used = json.loads(used_path.read_text())
    now = spans_as_json(spans)
    if used != now:
        print(f"warning: period spans changed since extract wrote {used_path}")
        print(f"  extract used: {used}")
        print(f"  now detected: {now}")
        print("  re-run extract (--force extract) so the possessions match these spans")


def step_halfcourt(game, paths, dry, allow_mismatch=False):
    if dry:
        print(f"+ half-court records per period -> {paths.halfcourt}")
        return
    summary, info, spans, period_fields = check_periods(game, paths, allow_mismatch)
    warn_if_spans_changed(paths, spans)
    reads_raw = sb.load_timeline(paths.scoreboard_raw)
    possessions = read_possessions(paths.trajectories)
    records = []
    coverage = {"game": game.espn_id, "layout": game.layout, **period_fields, "periods": []}
    trust: list[tuple[int, float, float]] = []
    for span in spans:
        events = pbp.parse_events(summary["plays"], period=span.period, team_by_id=info.team_by_id)
        reads = sb.clean_timeline([r for r in reads_raw if span.t_lo - 5 <= r.t <= span.t_hi + 5],
                                  valid_states=pbp.score_states(events))
        # span.period is an ESPN period number only because check_periods passed above.
        recs = H.build_records(game.espn_id, "Duke", events, reads, possessions,
                               period_length=pbp.period_length(span.period), period=span.period,
                               span=(span.t_lo, span.t_hi))
        records.extend(recs)
        dead = [r for r in recs if r.located and r.t0 is not None and not r.transition
                and r.start_type in ("ato", "dead")]
        clock_rate, score_rate = trust_rates(reads)
        trust.append((len(reads), clock_rate, score_rate))
        coverage["periods"].append({
            "period": span.period, "t_lo": span.t_lo, "t_hi": span.t_hi, "intervals": len(recs),
            "located": sum(r.located and r.t0 is not None for r in recs),
            "dead_ball": len(dead), "dead_ball_with_setup": sum(not r.no_setup for r in dead),
            "start_types": dict(Counter(r.start_type for r in recs)),
            "ocr_reads": len(reads), "clock_trust_rate": round(clock_rate, 3),
            "score_trust_rate": round(score_rate, 3),
        })
    coverage["ocr_reads"] = len(reads_raw)
    coverage.update(weighted_trust_rates(trust))
    coverage.update(espn_agreement(possessions))
    cost = paths.root / "cost.json"
    if cost.exists():
        coverage["cost_estimate"] = json.loads(cost.read_text()).get("estimate_usd")
    write_jsonl(paths.halfcourt, records)
    paths.coverage.write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))


def trust_rates(reads) -> tuple[float, float]:
    """Fractions of already-cleaned `reads` whose clock, and whose score, survived cleaning.

    Must be given one period's reads at a time: `sb.clean_timeline` enforces a non-increasing
    clock, so cleaning a whole game in one pass rejects nearly every read after the clock resets
    at half time (0.41 clock trust for the Michigan game, against 0.85/0.87 per period).
    """
    n = len(reads)
    if not n:
        return 0.0, 0.0
    clock = sum(r.clock is not None for r in reads) / n
    score = sum(r.away is not None and r.home is not None for r in reads) / n
    return clock, score


def weighted_trust_rates(per_period: list[tuple[int, float, float]]) -> dict:
    """Game-level trust rates: the per-period rates weighted by each period's read count."""
    total = sum(n for n, _, _ in per_period)
    if not total:
        return {"clock_trust_rate": 0.0, "score_trust_rate": 0.0}
    return {
        "clock_trust_rate": round(sum(n * c for n, c, _ in per_period) / total, 3),
        "score_trust_rate": round(sum(n * s for n, _, s in per_period) / total, 3),
    }


def espn_agreement(possessions) -> dict:
    """Possessions where ESPN's points and the scoreboard delta both exist, and agree or not."""
    both = [p for p in possessions
            if p.points_scored is not None and p.scoreboard_points is not None]
    agree = sum(p.points_scored == p.scoreboard_points for p in both)
    return {"espn_agree": agree, "espn_disagree": len(both) - agree}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("espn_id")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--force", default="",
                     help="comma-separated steps to redo even if output exists; every step "
                          "downstream of one of these is redone too")
    ap.add_argument("--allow-period-mismatch", action="store_true",
                     help="continue when the detected period count disagrees with ESPN's")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=10.0)
    args = ap.parse_args()
    game = get_game(args.espn_id)
    paths = GamePaths.for_game(game.espn_id)
    asked = set(args.force.split(",")) - {""}
    unknown = asked - set(DOWNSTREAM)
    if unknown:
        raise SystemExit(f"unknown --force step(s) {', '.join(sorted(unknown))}; "
                         f"choose from {', '.join(STEPS)}")
    force = expand_force(asked)
    if force:
        print(f"force: {', '.join(s for s in STEPS if s in force)} "
              f"(from {', '.join(s for s in STEPS if s in asked)})")
    # annotate has no single output file to check: its output is trajectories.jsonl, which
    # already exists after extract, so it always runs (cheap, uses --skip-ocr).
    outputs = {"download": paths.video, "gpu": paths.raw_dir / "frames_00.jsonl",
               "extract": paths.trajectories, "ocr": paths.scoreboard_raw,
               "espn": paths.espn_summary, "annotate": None, "halfcourt": paths.halfcourt}
    for step in args.steps.split(","):
        if step not in outputs:
            raise SystemExit(f"unknown step {step!r}; choose from {', '.join(STEPS)}")
        out = outputs[step]
        if out is not None and out.exists() and step not in force:
            print(f"skip {step}: {out} exists")
            continue
        print(f"== {step} ({game.opponent}, {game.date}, layout {game.layout})")
        if step == "download":
            step_download(game, paths, args.dry_run)
        elif step == "gpu":
            step_gpu(game, paths, args.dry_run, args.max_cost)
        elif step == "espn":
            step_espn(game, paths, args.dry_run)
        elif step == "extract":
            if not paths.espn_summary.exists() and not args.dry_run:
                step_espn(game, paths, False)
            step_extract(game, paths, args.dry_run, args.allow_period_mismatch)
        elif step == "ocr":
            step_ocr(game, paths, args.dry_run)
        elif step == "annotate":
            step_annotate(game, paths, args.dry_run, args.allow_period_mismatch)
        elif step == "halfcourt":
            step_halfcourt(game, paths, args.dry_run, args.allow_period_mismatch)
    if not args.dry_run:
        paths.game_json.write_text(json.dumps(game.__dict__, indent=2))


if __name__ == "__main__":
    main()
