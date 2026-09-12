"""Discover candidate sets by clustering per-possession feature vectors, with montages.

    uv run python scripts/cluster_plays.py \\
        --features data/plays/features.npz --index data/plays/features_index.json \\
        --records data/plays/halfcourt.jsonl \\
        --k-min 8 --k-max 30 --out data/plays/clusters.json --montages data/plays/montages

Fits `plays.fit_clusters` on the `split == "train"` rows that have a setup frame (`no_setup ==
False`) -- rows with one or two detected players and no setup frame carry almost no formation
signal and would otherwise pull the centroids toward detection noise. Every other row (no-setup
train rows, and all test/ncaa rows) is assigned to the nearest centroid afterwards via
`plays.assign` and is flagged `"assigned_only": true` in the output. Writes `clusters.json` and
renders a 3x3 grid of setup tiles (the nine nearest *fit* members) per cluster to
`--montages/cluster_<c>.png`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basketball_plays import montage as M
from basketball_plays import plays as P
from basketball_plays import zones as Z
from basketball_plays.features import FeatureRow
from basketball_plays.halfcourt import HalfcourtRecord


def load_records(path: str | Path) -> dict[str, HalfcourtRecord]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = HalfcourtRecord.from_dict(json.loads(line))
                out[f"{rec.game_id}:{rec.period}:{rec.index}"] = rec
    return out


def load_rows(features_path: str | Path, index_path: str | Path) -> list[FeatureRow]:
    data = np.load(features_path)
    X, H = data["X"], data["H"]
    meta = json.loads(Path(index_path).read_text())
    rows = meta["rows"]
    if len(rows) != len(X):
        raise ValueError(f"index has {len(rows)} rows but features has {len(X)}")
    return [
        FeatureRow(game_id=r["game_id"], period=r["period"], index=r["index"], bucket=r["bucket"],
                   split=r["split"], no_setup=r["no_setup"], vector=X[i], handler=H[i],
                   context=r.get("context", {}))
        for i, r in enumerate(rows)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/plays/features.npz")
    ap.add_argument("--index", default="data/plays/features_index.json")
    ap.add_argument("--records", default="data/plays/halfcourt.jsonl")
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--out", default="data/plays/clusters.json")
    ap.add_argument("--montages", default="data/plays/montages")
    args = ap.parse_args()

    rows = load_rows(args.features, args.index)
    X = np.stack([r.vector for r in rows])
    fit_mask = np.array([r.split == "train" and not r.no_setup for r in rows])
    fit_idx = np.where(fit_mask)[0]
    assign_idx = np.where(~fit_mask)[0]

    model = P.fit_clusters(X[fit_idx], k_range=range(args.k_min, args.k_max + 1))
    labels = np.empty(len(rows), dtype=int)
    labels[fit_idx] = model.labels_train
    if len(assign_idx):
        labels[assign_idx] = P.assign(model, X[assign_idx])
    clusters = P.summarize(model, rows, labels, Z.ZONE_NAMES, fit_mask=fit_mask)

    print("k sweep:", model.sweep)
    print(f"chosen k={model.k} silhouette={model.silhouette:.3f} "
          f"stability_ari={model.stability_ari:.3f} small_corpus={model.small_corpus} "
          f"n_fit={len(fit_idx)} n_assigned_only={len(assign_idx)}")
    for c in clusters:
        print(f"cluster {c['cluster']}: n={c['n']} n_fit={c['n_fit']} "
              f"by_bucket={c['n_by_bucket']} by_split={c['n_by_split']}")

    out = {
        "k": model.k, "silhouette": model.silhouette, "stability_ari": model.stability_ari,
        "small_corpus": model.small_corpus, "sweep": {str(k): v for k, v in model.sweep.items()},
        "n_fit": len(fit_idx), "n_assigned_only": len(assign_idx),
        "clusters": clusters,
        "labels": {f"{r.game_id}:{r.period}:{r.index}": {
            "cluster": int(labels[i]), "assigned_only": bool(not fit_mask[i]),
        } for i, r in enumerate(rows)},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    records = load_records(args.records)
    montages_dir = Path(args.montages)
    montages_dir.mkdir(parents=True, exist_ok=True)
    for stale in montages_dir.glob("cluster_*.png"):
        stale.unlink()
    for c in clusters:
        tiles = []
        for game_id, period, index, _distance in c["members"][:9]:
            rec = records.get(f"{game_id}:{period}:{index}")
            if rec is not None:
                tiles.append(M.render_setup_tile(rec))
        if not tiles:
            continue
        grid = M.grid(tiles, cols=3)
        cv2.imwrite(str(montages_dir / f"cluster_{c['cluster']}.png"), grid)

    print(f"wrote {out_path} and {len(clusters)} montage(s) to {montages_dir}")


if __name__ == "__main__":
    main()
