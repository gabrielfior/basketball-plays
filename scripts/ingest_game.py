"""Ingest one game end to end, resumably.

Steps (each skipped when its output exists):
  download   yt-dlp 720p avc1 video -> data/games/<id>/video.mp4
  espn       ESPN summary -> espn_summary.json
  gpu        Modal Stage A -> raw/frames_XX.jsonl (prints a cost estimate first)
  ocr        scoreboard timeline -> scoreboard_raw.jsonl (whole video; layout from games.json)
  extract    Stage B -> trajectories.jsonl (rosters from the ESPN summary; offense learned per
             period from the scoreboard timeline, not guessed from a time window)
  annotate   clock, score, outcome per possession, per period
  halfcourt  Duke half-court records per period -> halfcourt.jsonl, coverage.json

OCR now runs before extract (it no longer needs possession times to pick its range - it reads
the whole video) so extract can pass the scoreboard timeline's period spans straight in.

    uv run python scripts/ingest_game.py 401817238 --dry-run
    uv run python scripts/ingest_game.py 401817238
    uv run python scripts/ingest_game.py 401817238 --steps halfcourt --force halfcourt
"""

from __future__ import annotations

import argparse
import json
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


def run(cmd: list[str], dry: bool) -> None:
    print("+", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def step_download(game, paths, dry):
    paths.root.mkdir(parents=True, exist_ok=True)
    run(["uvx", "yt-dlp", "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
         "-o", str(paths.video), game.youtube_url], dry)


def step_gpu(game, paths, dry, max_cost):
    run(["uv", "run", "modal", "run", "modal_app.py", "--video", str(paths.video),
         "--game", game.espn_id, "--end", "-1", "--out-dir", str(paths.raw_dir),
         "--max-cost", str(max_cost)], dry)


def step_espn(game, paths, dry):
    if dry:
        print(f"+ fetch ESPN summary {game.espn_id} -> {paths.espn_summary}")
        return
    pbp.fetch_summary(game.espn_id, cache=paths.espn_summary)


def step_extract(game, paths, dry):
    run(["uv", "run", "python", "scripts/extract_trajectories.py", str(paths.video), "--skip-gpu",
         "--raw-dir", str(paths.raw_dir), "--out", str(paths.trajectories),
         "--rosters-from", str(paths.espn_summary),
         "--periods-from", str(paths.scoreboard_raw)], dry)


def step_ocr(game, paths, dry):
    if dry:
        print(f"+ scoreboard OCR layout={game.layout} -> {paths.scoreboard_raw}")
        return
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(paths.video)], capture_output=True, text=True, check=True)
    duration = float(probe.stdout.strip())
    raw = sb.read_timeline(str(paths.video), 0.0, duration + 1.0, every_s=1.0, layout=game.layout)
    sb.write_timeline(paths.scoreboard_raw, raw)


def step_annotate(game, paths, dry):
    if dry:
        print(f"+ annotate per period -> {paths.trajectories}")
        return
    summary = json.loads(paths.espn_summary.read_text())
    info = gameinfo.from_summary(summary)
    spans = P.period_spans(sb.load_timeline(paths.scoreboard_raw))
    for span in spans:
        run(["uv", "run", "python", "scripts/annotate_outcomes.py",
             str(paths.video), str(paths.trajectories), "--skip-ocr",
             "--raw-ocr", str(paths.scoreboard_raw), "--espn-cache", str(paths.espn_summary),
             "--period", str(span.period), "--t-lo", str(span.t_lo), "--t-hi", str(span.t_hi)], dry)
    print(f"annotated {len(spans)} periods for {info.home} vs {info.away}")


def step_halfcourt(game, paths, dry):
    if dry:
        print(f"+ half-court records per period -> {paths.halfcourt}")
        return
    summary = json.loads(paths.espn_summary.read_text())
    info = gameinfo.from_summary(summary)
    reads_raw = sb.load_timeline(paths.scoreboard_raw)
    possessions = read_possessions(paths.trajectories)
    records = []
    coverage = {"game": game.espn_id, "periods": []}
    for span in P.period_spans(reads_raw):
        events = pbp.parse_events(summary["plays"], period=span.period, team_by_id=info.team_by_id)
        reads = sb.clean_timeline([r for r in reads_raw if span.t_lo - 5 <= r.t <= span.t_hi + 5],
                                  valid_states=pbp.score_states(events))
        recs = H.build_records(game.espn_id, "Duke", events, reads, possessions,
                               period_length=pbp.period_length(span.period), period=span.period,
                               span=(span.t_lo, span.t_hi))
        records.extend(recs)
        dead = [r for r in recs if r.located and r.t0 is not None and not r.transition
                and r.start_type in ("ato", "dead")]
        coverage["periods"].append({
            "period": span.period, "t_lo": span.t_lo, "t_hi": span.t_hi, "intervals": len(recs),
            "located": sum(r.located and r.t0 is not None for r in recs),
            "dead_ball": len(dead), "dead_ball_with_setup": sum(not r.no_setup for r in dead),
            "start_types": dict(Counter(r.start_type for r in recs)),
        })
    write_jsonl(paths.halfcourt, records)
    paths.coverage.write_text(json.dumps(coverage, indent=2))
    print(json.dumps(coverage, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("espn_id")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--force", default="",
                     help="comma-separated steps to redo even if output exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-cost", type=float, default=10.0)
    args = ap.parse_args()
    game = get_game(args.espn_id)
    paths = GamePaths.for_game(game.espn_id)
    force = set(args.force.split(",")) - {""}
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
            step_extract(game, paths, args.dry_run)
        elif step == "ocr":
            step_ocr(game, paths, args.dry_run)
        elif step == "annotate":
            step_annotate(game, paths, args.dry_run)
        elif step == "halfcourt":
            step_halfcourt(game, paths, args.dry_run)
    if not args.dry_run:
        paths.game_json.write_text(json.dumps(game.__dict__, indent=2))


if __name__ == "__main__":
    main()
