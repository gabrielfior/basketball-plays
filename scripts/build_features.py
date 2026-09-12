"""Build per-record feature vectors from the aggregate half-court records file.

    uv run python scripts/build_features.py --records data/plays/halfcourt.jsonl --out-dir data/plays

Writes `features.npz` (X: cluster vectors, H: handler zones, S: snapshots stacked with NaN for
missing) and `features_index.json` (per-row metadata plus the roster order) into `--out-dir`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import features as F
from basketball_plays import gameinfo
from basketball_plays.games import load_registry
from basketball_plays.halfcourt import HalfcourtRecord


def load_records(path: str | Path) -> list[HalfcourtRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(HalfcourtRecord.from_dict(json.loads(line)))
    return records


def in_scope(rec: HalfcourtRecord) -> bool:
    return (rec.start_type in ("ato", "dead") and rec.located and not rec.transition
            and rec.t0 is not None)


def duke_roster(games_root: Path = Path("data/games")) -> list[str]:
    names: set[str] = set()
    for summary_path in sorted(games_root.glob("*/espn_summary.json")):
        summary = json.loads(summary_path.read_text())
        info = gameinfo.from_summary(summary)
        names.update(info.rosters.get("Duke", {}).values())
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/plays/halfcourt.jsonl")
    ap.add_argument("--out-dir", default="data/plays")
    args = ap.parse_args()

    records = [r for r in load_records(args.records) if in_scope(r)]
    splits_by_game = {g.espn_id: g.split for g in load_registry()}
    roster = duke_roster()
    rows = F.build_rows(records, splits_by_game, roster=roster)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.stack([r.vector for r in rows]) if rows else np.zeros((0, 5 * F.Z.N_ZONES + 5 + 3))
    Hh = np.stack([r.handler for r in rows]) if rows else np.zeros((0, len(F.BINS)), dtype=int)
    if rows and rows[0].snapshot is not None:
        S = np.stack([r.snapshot for r in rows])
    else:
        S = np.full((len(rows), len(roster), F.Z.N_ZONES + 1), np.nan)

    np.savez_compressed(out_dir / "features.npz", X=X, H=Hh, S=S)

    index = {
        "roster": roster,
        "rows": [
            {"game_id": r.game_id, "period": r.period, "index": r.index, "bucket": r.bucket,
             "split": r.split, "no_setup": r.no_setup, "context": r.context}
            for r in rows
        ],
    }
    (out_dir / "features_index.json").write_text(json.dumps(index, indent=2))

    bucket_counts = Counter(r.bucket for r in rows)
    split_counts = Counter(r.split for r in rows)
    print(f"wrote {len(rows)} rows to {out_dir}")
    print("buckets:", dict(bucket_counts))
    print("splits:", dict(split_counts))


if __name__ == "__main__":
    main()
