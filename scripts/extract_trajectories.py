"""Generate trajectories.jsonl (possessions with player and ball court trajectories) from a video.

Stage A (GPU, Modal) writes per-frame detections to data/games/<id>/raw/ (or data/raw/ for the
legacy default game); Stage B (local) turns them into possessions. Run both, for one game in the
27-game set:

    uv run python scripts/extract_trajectories.py data/games/401817238/video.mp4 --game 401817238 \\
        --rosters-from data/games/401817238/espn_summary.json

Or the original single-game (legacy) invocation, still supported with `--game` left at its
"default" value:

    uv run python scripts/extract_trajectories.py data/duke_michigan_q1.mp4 --end 2135 --out data/trajectories.jsonl

Re-run only Stage B on existing raw detections:

    uv run python scripts/extract_trajectories.py data/duke_michigan_q1.mp4 --skip-gpu --out data/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import gameinfo, periods, scoreboard
from basketball_plays.extract import build_possessions
from basketball_plays.games import GamePaths
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import read_frames, write_jsonl


def run_gpu_stage(video: str, game: str, start: float, end: float, fps: float, raw_dir: str,
                   skip_upload: bool) -> None:
    cmd = ["uv", "run", "modal", "run", "modal_app.py", "--video", video, "--game", game,
           "--start", str(start), "--end", str(end), "--fps", str(fps), "--out-dir", raw_dir]
    if skip_upload:
        cmd.append("--skip-upload")
    env = {**os.environ, "MODAL_IMAGE_BUILDER_VERSION": os.environ.get("MODAL_IMAGE_BUILDER_VERSION", "2025.06")}
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=Path(__file__).resolve().parents[1])


DEFAULT_END = 2135.0  # legacy default (35:35), overridden to the whole video when --game is set


def resolve_team_map(spec: str, home: str, away: str) -> dict[int, str] | None:
    """Turn a `--team-map` spec like `"0=Duke"` into `{0: 'Duke', 1: <the other team>}`.

    `auto` (the default) returns None, leaving the cluster-to-team decision to the appearance
    and jersey evidence. The named team must be one of this game's two teams, so a typo or a
    team from another game fails here instead of silently mislabelling every possession.
    """
    if spec == "auto":
        return None
    cluster_text, _, name = spec.partition("=")
    cluster_text, name = cluster_text.strip(), name.strip()
    if cluster_text not in ("0", "1") or not name:
        sys.exit(f"--team-map must look like '0={home}' or '1={away}' (got {spec!r})")
    match = next((t for t in (home, away) if t.lower() == name.lower()), None)
    if match is None:
        sys.exit(f"--team-map team {name!r} is not one of this game's teams "
                 f"({home!r}, {away!r})")
    cluster = int(cluster_text)
    other = away if match == home else home
    return {cluster: match, 1 - cluster: other}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--game", default="default",
                    help="ESPN game id; defaults --raw-dir/--out to data/games/<id>/... "
                         "('default' keeps the legacy data/raw, data/trajectories.jsonl paths)")
    ap.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    ap.add_argument("--end", type=float, default=DEFAULT_END, help="clip end in seconds (35:35)")
    ap.add_argument("--fps", type=float, default=10.0, help="sampling rate for perception")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rosters-from", help="path to an espn_summary.json for rosters and home/away")
    ap.add_argument("--periods-from",
                    help="path to a scoreboard_raw.jsonl; learns one offense map per period "
                         "instead of guessing from a time window")
    ap.add_argument("--allow-no-periods", action="store_true",
                    help="with --periods-from, continue (falling back to the time-windowed "
                         "offense rule) when no period spans could be detected")
    ap.add_argument("--skip-gpu", action="store_true", help="reuse raw detections in --raw-dir")
    ap.add_argument("--skip-upload", action="store_true", help="video already on the Modal volume")
    ap.add_argument("--team-map", default="auto",
                    help="force a cluster's team, e.g. \"0=Duke\"; the team must be one of this "
                         "game's two teams (from --rosters-from, else Duke/Michigan) and the "
                         "other cluster gets the other team")
    ap.add_argument("--min-players", type=int, default=6)
    ap.add_argument("--min-duration", type=float, default=3.0)
    ap.add_argument("--max-gap", type=float, default=3.0, help="invalid-view gap that ends a possession")
    ap.add_argument("--merge-gap", type=float, default=12.0,
                    help="rejoin same-half runs separated by at most this many seconds (replays)")
    args = ap.parse_args()

    is_legacy = args.game == "default"
    game_paths = None if is_legacy else GamePaths.for_game(args.game)
    raw_dir_str = args.raw_dir or (str(game_paths.raw_dir) if game_paths else "data/raw")
    out = args.out or (str(game_paths.trajectories) if game_paths else "data/trajectories.jsonl")

    if not args.skip_gpu:
        # An explicit `--end 2135` is indistinguishable from the untouched default and is also
        # treated as "use the whole video" once a real --game is given.
        gpu_end = -1.0 if (not is_legacy and args.end == DEFAULT_END) else args.end
        run_gpu_stage(args.video, args.game, args.start, gpu_end, args.fps,
                      raw_dir_str, args.skip_upload)

    raw_dir = Path(raw_dir_str)
    frames = []
    for p in sorted(raw_dir.glob("frames_*.jsonl")):
        frames.extend(read_frames(p))
    if not frames:
        sys.exit(f"no frames_*.jsonl in {raw_dir}")
    meta = json.loads((raw_dir / "meta.json").read_text()) if (raw_dir / "meta.json").exists() else {}
    brightness = {int(k): float(v) for k, v in meta.get("cluster_brightness", {}).items()}
    info = None
    if args.rosters_from:
        info = gameinfo.from_summary(json.loads(Path(args.rosters_from).read_text()))
    team_map = resolve_team_map(args.team_map, info.home if info else DUKE,
                                info.away if info else MICHIGAN)

    period_spans = None
    if args.periods_from:
        spans = periods.period_spans(scoreboard.load_timeline(args.periods_from))
        if not spans:
            # An empty span list means the scoreboard timeline had no usable clock reads: the
            # offense maps would silently fall back to the one-window rule for the whole game.
            print(f"no period spans detected in {args.periods_from}: the scoreboard timeline has "
                  "no usable clock reads (wrong layout, or OCR failed)")
            if not args.allow_no_periods:
                sys.exit("re-run the ocr step, or pass --allow-no-periods to fall back to the "
                         "time-windowed offense rule")
            print("falling back to the time-windowed offense rule (--allow-no-periods)")
        period_spans = [(s.t_lo, s.t_hi) for s in spans]

    fps = float(meta.get("fps", args.fps))
    possessions = build_possessions(frames, fps=fps, cluster_brightness=brightness, team_map=team_map,
                                    min_players=args.min_players, min_duration=args.min_duration,
                                    max_gap=args.max_gap, merge_same_half_gap=args.merge_gap,
                                    rosters=info.rosters if info else None,
                                    home_team=info.home if info else DUKE,
                                    away_team=info.away if info else MICHIGAN,
                                    period_spans=period_spans)
    write_jsonl(out, possessions)
    total = sum(p.end_time - p.start_time for p in possessions)
    print(f"{len(frames)} frames -> {len(possessions)} possessions ({total:.0f}s of play) -> {out}")
    for p in possessions[:10]:
        named = sum(1 for q in p.players if q.name)
        print(f"  #{p.possession_id:3d} {p.start_time:7.1f}-{p.end_time:7.1f}s  {p.offense_team or '?':9s} -> "
              f"{p.attacking_basket:5s} players={len(p.players):2d} named={named} ball_pts={len(p.ball)}")


if __name__ == "__main__":
    main()
