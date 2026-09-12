"""Set discovery: standardise, PCA, k-means over a k sweep chosen by a one-standard-error rule on
the silhouette, stability by adjusted Rand index across seeds, nearest-centroid assignment for
held-out rows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

#: The sweep window scales continuously with the corpus instead of flipping at a threshold:
#: `k_min` is always 4 and `k_max` is `n // 12` clipped into `[K_MAX_LO, K_MAX_HI]`. A hard gate
#: used to move the window by eight k at a single extra row, which made the chosen k jump for
#: reasons that had nothing to do with the data.
K_MIN = 4
K_MAX_LO, K_MAX_HI = 6, 30
ROWS_PER_K = 12

#: `small_corpus` is a graded advisory, not a switch: below this many fit rows the clusters are
#: provisional, and the page says so.
ADVISORY_ROWS = 400

STABILITY_SEEDS = (1, 2, 3, 4)


@dataclass
class ClusterModel:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    k: int
    #: mean silhouette at the chosen k; `None` when the corpus was too small to sweep
    silhouette: float | None
    stability_ari: float
    labels_train: np.ndarray
    small_corpus: bool = False
    #: k -> {"mean": mean silhouette, "se": std / sqrt(n)}
    sweep: dict = field(default_factory=dict)
    #: the k with the best mean silhouette, before the 1-SE rule preferred a smaller one
    k_best_raw: int | None = None


def _project(scaler, pca, X):
    return pca.transform(scaler.transform(X))


def sweep_range(n: int, k_range: range) -> range:
    """The k sweep for `n` rows, clamped by the caller's `k_range`.

    The window is `K_MIN .. clip(n // ROWS_PER_K, K_MAX_LO, K_MAX_HI)`. The caller's range only
    ever narrows it: the lower bound is never raised and the upper bound is never widened.
    """
    k_max = int(np.clip(n // ROWS_PER_K, K_MAX_LO, K_MAX_HI))
    lo, hi = min(k_range.start, K_MIN), min(k_range.stop, k_max + 1)
    return range(lo, max(hi, lo))


def _choose_k(sweep: dict[int, dict]) -> tuple[int, int]:
    """(k by the one-standard-error rule, k with the best mean silhouette).

    The 1-SE rule takes the smallest k whose mean silhouette is within one standard error of the
    best mean -- with a few hundred possessions the sweep's peak is well inside the noise, so
    picking the raw argmax reads structure into sampling error. Fewer, larger sets are also the
    more useful answer when a human has to name them.
    """
    k_best = max(sweep, key=lambda k: sweep[k]["mean"])
    threshold = sweep[k_best]["mean"] - sweep[k_best]["se"]
    return min(k for k, v in sweep.items() if v["mean"] >= threshold), k_best


def fit_clusters(X_train: np.ndarray, k_range=range(8, 31), n_components: int = 20,
                 seed: int = 0) -> ClusterModel:
    n = len(X_train)
    if n < 3:
        raise ValueError(f"cannot cluster {n} row(s): at least 3 are needed for k=2 with a "
                         f"silhouette")
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=min(n_components, n - 1, X_train.shape[1]), random_state=seed)
    Z = pca.fit_transform(scaler.transform(X_train))
    sweep: dict[int, dict] = {}
    for k in sweep_range(n, k_range):
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        s = silhouette_samples(Z, km.labels_)
        sweep[k] = {"mean": float(np.mean(s)), "se": float(np.std(s) / np.sqrt(len(s)))}
    if sweep:
        k, k_best_raw = _choose_k(sweep)
        silhouette = sweep[k]["mean"]
    else:
        # Every k in the window was >= n (a corpus of a handful of rows): k=2 is all that is left.
        k, k_best_raw, silhouette = 2, None, None
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    aris = [adjusted_rand_score(km.labels_,
                                KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z).labels_)
            for s in STABILITY_SEEDS]
    return ClusterModel(scaler, pca, km, k, silhouette, float(np.mean(aris)), km.labels_,
                        n < ADVISORY_ROWS, sweep, k_best_raw)


def assign(model: ClusterModel, X: np.ndarray) -> np.ndarray:
    return model.kmeans.predict(_project(model.scaler, model.pca, X))


def summarize(model: ClusterModel, rows, labels: np.ndarray, zone_names: list[str],
              n_bins: int = 5, fit_mask: np.ndarray | None = None) -> list[dict]:
    """Per-cluster summary. `fit_mask` marks which `rows` shaped the centroids (all of them when
    `None`); `n` stays the total membership while `n_fit` counts only the fit ones.

    `centroid_occupancy`, `top_zones`, `handler_path` and `members` (the montage tiles) all
    describe the FIT members only, so the card matches the centroid and the tiles beside it
    rather than averaging in possessions that were merely snapped to the nearest centroid.
    """
    if fit_mask is None:
        fit_mask = np.ones(len(rows), dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)
    Z = _project(model.scaler, model.pca, np.stack([r.vector for r in rows]))
    out = []
    for c in range(model.k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        idx_fit = idx[fit_mask[idx]]
        # A cluster with no fit members cannot happen on the fit path (k-means labels every
        # centroid), but fall back to the full membership rather than averaging nothing.
        src = idx_fit if len(idx_fit) else idx
        occ = np.mean([rows[i].vector[: n_bins * len(zone_names)] for i in src], axis=0)
        occ = occ.reshape(n_bins, len(zone_names))
        dists = np.linalg.norm(Z[idx_fit] - model.kmeans.cluster_centers_[c], axis=1)
        order = idx_fit[np.argsort(dists)]
        top = [zone_names[j] for j in np.argsort(-occ[1])[:5]]
        path = []
        for b in range(n_bins):
            hs = [int(rows[i].handler[b]) for i in src if rows[i].handler[b] >= 0]
            path.append(zone_names[int(np.bincount(hs).argmax())] if hs else "-")
        out.append({
            "cluster": int(c), "n": len(idx), "n_fit": len(idx_fit),
            "n_by_bucket": {b: int(sum(1 for i in idx if rows[i].bucket == b))
                            for b in ("ato", "inbound", "after_score")},
            "n_by_split": {s: int(sum(1 for i in idx if rows[i].split == s))
                           for s in ("train", "test", "ncaa")},
            "centroid_occupancy": occ.round(3).tolist(), "top_zones": top, "handler_path": path,
            "members": [(rows[i].game_id, rows[i].period, rows[i].index,
                         round(float(np.linalg.norm(Z[i] - model.kmeans.cluster_centers_[c])), 3))
                        for i in order],
        })
    return out
