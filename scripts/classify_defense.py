"""Classify the opponent's defensive scheme (man vs. zone) per Duke half-court possession.

    uv run python scripts/classify_defense.py --records data/plays/halfcourt.jsonl \\
        --out data/plays/defense.json

Writes `{"<game_id>:<period>:<index>": {"defense": ..., "confidence": ..., "features": {...}}}`
and prints label counts overall and per game.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import defense as D
from basketball_plays.features import in_scope
from basketball_plays.halfcourt import HalfcourtRecord


def load_records(path: str | Path) -> list[HalfcourtRecord]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(HalfcourtRecord.from_dict(json.loads(line)))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="data/plays/halfcourt.jsonl")
    ap.add_argument("--out", default="data/plays/defense.json")
    args = ap.parse_args()

    records = [r for r in load_records(args.records) if in_scope(r)]
    feats = [D.features(r) for r in records]
    labels = D.label_rule(feats)

    out: dict[str, dict] = {}
    for rec, f, (label, conf) in zip(records, feats, labels):
        key = f"{rec.game_id}:{rec.period}:{rec.index}"
        out[key] = {"defense": label, "confidence": conf, "features": f}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    overall = Counter(v["defense"] for v in out.values())
    per_game: dict[str, Counter] = defaultdict(Counter)
    for rec, (label, _) in zip(records, labels):
        per_game[rec.game_id][label] += 1

    print(f"wrote {len(out)} records to {out_path}")
    print("overall:", dict(overall))
    print("per game:")
    for game_id in sorted(per_game):
        print(f"  {game_id}: {dict(per_game[game_id])}")


if __name__ == "__main__":
    main()
