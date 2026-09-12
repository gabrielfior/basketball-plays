# scripts/ingest_all.py
"""Ingest every registered game that has no half-court records yet, in a safe order.

    uv run python scripts/ingest_all.py --dry-run
    uv run python scripts/ingest_all.py --layouts cbs,ncaa,cw,cbssn --max-games 4  # one/layout
    uv run python scripts/ingest_all.py --split train --prune-video
    uv run python scripts/ingest_all.py --wave1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import gameinfo
from basketball_plays.games import Game, GamePaths, load_registry

FIRST_PER_LAYOUT = {
    "cbs": "401820644", "ncaa": "401856478", "cw": "401820689", "cbssn": "401817230",
}
# Wave 1 of the phase-1B batch: Indiana State, then the Florida State (cbs) and Siena (ncaa)
# layout games so a layout bug surfaces before the big ESPN-layout run.
WAVE1 = ["401817231", "401820644", "401856478"]
# With --prune-video, the peak per-game footprint (one video plus its raw/derived outputs
# before the video is deleted) is about 2.5 GB, so 10 GB free is a safe floor; --min-free-gb
# overrides it for a run without pruning, or a smaller disk.
MIN_FREE_GB = 10
DEFAULT_GAME_TIMEOUT = 4 * 3600
PROGRESS_EVERY_S = 60
COST_RE = re.compile(r"estimated cost: .* = \$([0-9.]+)")


def ordered(games: list[Game]) -> list[Game]:
    """Layout-first games first (one per non-ESPN layout), then by split, then by date.

    This is the safe default order: it front-loads one game per non-ESPN scoreboard layout
    (cbs/ncaa/cw/cbssn) so a layout bug surfaces early, then works through the ESPN-layout train
    games by date, then test, then the NCAA tournament games. Use --only (or --wave1) to run a
    specific batch rather than reordering this.
    """
    first = [g for g in games if FIRST_PER_LAYOUT.get(g.layout) == g.espn_id]
    rest = [g for g in games if g not in first]
    rank = {"train": 0, "test": 1, "ncaa": 2}
    rest.sort(key=lambda g: (rank[g.split], g.date))
    return first + rest


def home_check(game: Game, summary: dict) -> str:
    """Compare the registry's `home` flag against ESPN's homeAway for Duke.

    "ok" when they agree, "mismatch" when they disagree (a likely wrong youtube_id/espn_id
    pairing, worth checking before spending GPU on the game), "unknown" when `summary` doesn't
    identify a Duke side at all, or is too malformed to parse. A malformed payload is exactly
    what a wrong youtube_id/espn_id pairing can produce, so this must never raise.
    """
    try:
        info = gameinfo.from_summary(summary)
        if info.home == "Duke":
            espn_home = True
        elif info.away == "Duke":
            espn_home = False
        else:
            return "unknown"
        return "ok" if espn_home == game.home else "mismatch"
    except Exception:  # noqa: BLE001 - a malformed summary (wrong id pairing) must not crash
        return "unknown"


def merge_status(old: dict | None, new: dict) -> dict:
    """Merge one attempt's fields (`new`) into a game's previous status record (`old`).

    A retry whose GPU step is skipped (already done, so no "estimated cost" line) yields
    `cost: None` for that attempt; naively overwriting the record would erase a real prior
    spend and under-count the running total that gates `--max-total-cost`. So: keep the last
    known non-None `cost`, and append the attempt's returncode/seconds/cost to a running
    `attempts` list instead of keeping only the most recent one. A `new` with no `"returncode"`
    (e.g. a home/away mismatch recorded without ever running a subprocess) merges its other
    fields (such as `home_check`) without touching `cost` or `attempts`.
    """
    old = old or {}
    merged = {**old, **new}
    if "returncode" not in new:
        merged["cost"] = old.get("cost")
        merged["attempts"] = old.get("attempts", [])
        return merged
    attempts = list(old.get("attempts", []))
    attempts.append({"returncode": new.get("returncode"), "seconds": new.get("seconds"),
                      "cost": new.get("cost")})
    merged["attempts"] = attempts
    merged["cost"] = new.get("cost") if new.get("cost") is not None else old.get("cost")
    return merged


def _write_status(status_path: Path, status: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))


def run_streamed(cmd: list[str], log_path: Path, timeout_s: float) -> tuple[int | str, int]:
    """Run `cmd`, streaming its combined stdout/stderr straight to `log_path`.

    Polls the child every second so it can print a one-line progress note every
    `PROGRESS_EVERY_S` seconds while waiting, and enforces `timeout_s`: past it, the child is
    killed and the returncode is the string "timeout". Returns `(returncode, elapsed_seconds)`.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    next_ping = t0 + PROGRESS_EVERY_S
    with log_path.open("w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        while True:
            try:
                rc = proc.wait(timeout=1)
                return rc, round(time.time() - t0)
            except subprocess.TimeoutExpired:
                now = time.time()
                if now - t0 > timeout_s:
                    proc.kill()
                    proc.wait()
                    return "timeout", round(now - t0)
                if now >= next_ping:
                    print(f"  ... still running ({round(now - t0)}s elapsed)")
                    next_ping = now + PROGRESS_EVERY_S


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default=None, help="comma-separated layouts to include")
    ap.add_argument("--split", default=None, help="train | test | ncaa")
    ap.add_argument("--only", default=None, help="comma-separated espn ids")
    ap.add_argument("--wave1", action="store_true",
                     help=f"shortcut for --only {','.join(WAVE1)}")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--max-total-cost", type=float, default=160.0)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                     help="stop before starting a game below this much free disk")
    ap.add_argument("--game-timeout", type=float, default=DEFAULT_GAME_TIMEOUT,
                     help="seconds before a stuck ingest_game.py subprocess is killed")
    ap.add_argument("--prune-video", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.wave1:
        if args.only:
            print("warning: both --wave1 and --only given; using --only")
        else:
            args.only = ",".join(WAVE1)

    games = ordered(load_registry())
    if args.layouts:
        games = [g for g in games if g.layout in args.layouts.split(",")]
    if args.split:
        games = [g for g in games if g.split == args.split]
    if args.only:
        games = [g for g in games if g.espn_id in args.only.split(",")]

    status_path = Path("data/games/status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    total = sum(v.get("cost", 0) or 0 for v in status.values())
    done = 0
    for g in games:
        paths = GamePaths.for_game(g.espn_id)
        if paths.halfcourt.exists():
            print(f"skip {g.espn_id} {g.opponent}: done")
            continue
        if args.max_games is not None and done >= args.max_games:
            break
        free_gb = shutil.disk_usage(".").free / 1e9
        if free_gb < args.min_free_gb:
            print(f"stop: only {free_gb:.1f} GB free")
            break
        if total >= args.max_total_cost:
            print(f"stop: running cost ${total:.2f} reached --max-total-cost")
            break

        cmd = ["uv", "run", "python", "scripts/ingest_game.py", g.espn_id, "--max-cost", "10"]
        print(f"== {g.espn_id} {g.opponent} {g.date} layout={g.layout} "
              f"(running total ${total:.2f})")
        if args.dry_run:
            print("+", " ".join(cmd))
            done += 1
            continue

        # Fetch (or reuse) the ESPN summary before spending GPU money, so a wrong
        # youtube_id/espn_id pairing can be caught first.
        if not paths.espn_summary.exists():
            espn_cmd = ["uv", "run", "python", "scripts/ingest_game.py", g.espn_id,
                        "--steps", "espn"]
            print("+", " ".join(espn_cmd))
            espn_proc = subprocess.run(espn_cmd, capture_output=True, text=True, check=False)
            if espn_proc.returncode != 0 or not paths.espn_summary.exists():
                out = espn_proc.stdout + espn_proc.stderr
                status[g.espn_id] = merge_status(status.get(g.espn_id), {
                    "returncode": espn_proc.returncode, "cost": None, "seconds": None,
                    "tail": out.splitlines()[-20:], "home_check": "unknown",
                })
                _write_status(status_path, status)
                print(f"FAILED {g.espn_id} fetching the ESPN summary "
                      f"(rc={espn_proc.returncode}); continuing")
                continue

        check = home_check(g, json.loads(paths.espn_summary.read_text()))
        if check == "mismatch":
            print(f"WARNING: home/away mismatch for {g.espn_id} ({g.opponent}, {g.date}); "
                  "likely wrong youtube_id/espn_id pairing; skipping GPU spend until resolved")
            status[g.espn_id] = merge_status(status.get(g.espn_id), {"home_check": check})
            _write_status(status_path, status)
            continue

        log_path = paths.root / "ingest.log"
        print(f"log: {log_path}")
        print("+", " ".join(cmd))
        rc, elapsed = run_streamed(cmd, log_path, args.game_timeout)
        out = log_path.read_text() if log_path.exists() else ""
        m = COST_RE.search(out)
        cost = float(m.group(1)) if m else None
        total += cost or 0
        status[g.espn_id] = merge_status(status.get(g.espn_id), {
            "returncode": rc, "cost": cost, "seconds": elapsed,
            "tail": out.splitlines()[-20:], "home_check": check,
        })
        _write_status(status_path, status)
        print(out.splitlines()[-1] if out.strip() else "(no output)")
        if rc != 0:
            print(f"FAILED {g.espn_id} (rc={rc}); continuing")
            continue
        if args.prune_video and paths.halfcourt.exists() and paths.video.exists():
            paths.video.unlink()
            print(f"pruned {paths.video}")
        done += 1
    print(f"finished: {done} games this run, running GPU estimate total ${total:.2f}")


if __name__ == "__main__":
    main()
