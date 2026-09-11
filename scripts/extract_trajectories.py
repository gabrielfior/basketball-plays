"""Generate trajectories.jsonl (possessions with player and ball court trajectories) from a video.

Stage A (GPU, Modal) writes per-frame detections to data/raw/; Stage B (local) turns them into
possessions. Run both:

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

from basketball_plays.extract import build_possessions
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import read_frames, write_jsonl


def run_gpu_stage(video: str, start: float, end: float, fps: float, raw_dir: str, skip_upload: bool) -> None:
    cmd = ["uv", "run", "modal", "run", "modal_app.py", "--video", video, "--start", str(start), "--end", str(end),
           "--fps", str(fps), "--out-dir", raw_dir]
    if skip_upload:
        cmd.append("--skip-upload")
    env = {**os.environ, "MODAL_IMAGE_BUILDER_VERSION": os.environ.get("MODAL_IMAGE_BUILDER_VERSION", "2025.06")}
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=Path(__file__).resolve().parents[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="clip start in seconds")
    ap.add_argument("--end", type=float, default=2135.0, help="clip end in seconds (35:35)")
    ap.add_argument("--fps", type=float, default=10.0, help="sampling rate for perception")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/trajectories.jsonl")
    ap.add_argument("--skip-gpu", action="store_true", help="reuse raw detections in --raw-dir")
    ap.add_argument("--skip-upload", action="store_true", help="video already on the Modal volume")
    ap.add_argument("--team-map", choices=["auto", "0=Duke", "0=Michigan"], default="auto",
                    help="force which appearance cluster is Duke")
    ap.add_argument("--min-players", type=int, default=6)
    ap.add_argument("--min-duration", type=float, default=3.0)
    ap.add_argument("--max-gap", type=float, default=3.0)
    args = ap.parse_args()

    if not args.skip_gpu:
        run_gpu_stage(args.video, args.start, args.end, args.fps, args.raw_dir, args.skip_upload)

    raw_dir = Path(args.raw_dir)
    frames = []
    for p in sorted(raw_dir.glob("frames_*.jsonl")):
        frames.extend(read_frames(p))
    if not frames:
        sys.exit(f"no frames_*.jsonl in {raw_dir}")
    meta = json.loads((raw_dir / "meta.json").read_text()) if (raw_dir / "meta.json").exists() else {}
    brightness = {int(k): float(v) for k, v in meta.get("cluster_brightness", {}).items()}
    team_map = None
    if args.team_map == "0=Duke":
        team_map = {0: DUKE, 1: MICHIGAN}
    elif args.team_map == "0=Michigan":
        team_map = {0: MICHIGAN, 1: DUKE}

    fps = float(meta.get("fps", args.fps))
    possessions = build_possessions(frames, fps=fps, cluster_brightness=brightness, team_map=team_map,
                                    min_players=args.min_players, min_duration=args.min_duration,
                                    max_gap=args.max_gap)
    write_jsonl(args.out, possessions)
    total = sum(p.end_time - p.start_time for p in possessions)
    print(f"{len(frames)} frames -> {len(possessions)} possessions ({total:.0f}s of play) -> {args.out}")
    for p in possessions[:10]:
        named = sum(1 for q in p.players if q.name)
        print(f"  #{p.possession_id:3d} {p.start_time:7.1f}-{p.end_time:7.1f}s  {p.offense_team or '?':9s} -> "
              f"{p.attacking_basket:5s} players={len(p.players):2d} named={named} ball_pts={len(p.ball)}")


if __name__ == "__main__":
    main()
