"""Refresh the aggregate half-court records, features, clusters, defence labels and the page.

Runs, in order:

  1. coverage_report.py   aggregate data/games/*/halfcourt.jsonl -> data/plays/halfcourt.jsonl,
                          coverage.md
  2. build_features.py    per-record feature vectors -> features.npz, features_index.json
  3. cluster_plays.py     candidate sets by clustering -> clusters.json, montages/ (skipped by
                          --skip-cluster)
  4. classify_defense.py  man/zone per possession -> defense.json
  5. build_page.py        the offline labelling page -> page/index.html

Each step runs as `uv run python scripts/<step>.py` with its defaults, so it reads and writes
under `data/plays/` the same way it would run standalone.

Re-clustering after adding games renumbers the clusters: `cluster_plays.py` refits KMeans on the
new, larger set of rows, so cluster 3 in the old run is not cluster 3 in the new one. Any names
already given to clusters on the labelling page are keyed by that id
(`labels.json`'s `"clusters"` map), so they would silently point at the wrong set after a plain
re-cluster. Before running this script without `--skip-cluster` on a corpus that has grown,
export the current decisions from the page's Export tab and save them to `data/plays/labels.json`
first -- `labels.py` keeps names by id, not by content, so nothing here re-associates them
automatically. A future task may add centroid matching to carry names across a re-cluster.

Use `--skip-cluster` to keep the existing `clusters.json` and `montages/` untouched (for example
while naming clusters, so ids already labelled do not move under you): the four other steps still
run, and the page re-reads the untouched clusters file plus the current `labels.json`.

    uv run python scripts/refresh_plays.py
    uv run python scripts/refresh_plays.py --skip-cluster
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("coverage_report", ["scripts/coverage_report.py"]),
    ("build_features", ["scripts/build_features.py"]),
    ("cluster_plays", ["scripts/cluster_plays.py"]),
    ("classify_defense", ["scripts/classify_defense.py"]),
    ("build_page", ["scripts/build_page.py"]),
]

PAGE_OUT = ROOT / "data" / "plays" / "page" / "index.html"


def run_step(name: str, args: list[str]) -> None:
    cmd = ["uv", "run", "python", *args]
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-cluster", action="store_true",
                     help="keep the existing clusters.json and montages/ (cluster ids stay put)")
    args = ap.parse_args(argv)

    for name, cmd_args in STEPS:
        if name == "cluster_plays" and args.skip_cluster:
            print("== cluster_plays: skipped (--skip-cluster), keeping clusters.json and montages/",
                  flush=True)
            continue
        run_step(name, cmd_args)

    size_mb = PAGE_OUT.stat().st_size / 1e6
    print(f"page: {PAGE_OUT} ({size_mb:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
