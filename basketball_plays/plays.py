"""Set discovery: standardise, PCA, k-means over a k sweep chosen by silhouette, stability by
adjusted Rand index across seeds, nearest-centroid assignment for held-out rows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SMALL_CORPUS = 150
STABILITY_SEEDS = (1, 2, 3, 4)


@dataclass
class ClusterModel:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    k: int
    silhouette: float
    stability_ari: float
    labels_train: np.ndarray
    small_corpus: bool = False
    sweep: dict = field(default_factory=dict)  # k -> silhouette


def _project(scaler, pca, X):
    return pca.transform(scaler.transform(X))


def fit_clusters(X_train: np.ndarray, k_range=range(8, 31), n_components: int = 20,
                 seed: int = 0) -> ClusterModel:
    n = len(X_train)
    small = n < SMALL_CORPUS
    if small:
        # Cap the sweep to what a small corpus can support, without raising a caller-supplied
        # lower bound (a test asking for a tiny k_range on a tiny corpus should keep it).
        cap = max(5, min(13, n // 12 + 1))
        lo, hi = min(k_range.start, 4), min(k_range.stop, cap)
        k_range = range(lo, hi if hi > lo else lo + 1)
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=min(n_components, n - 1, X_train.shape[1]), random_state=seed)
    Z = pca.fit_transform(scaler.transform(X_train))
    sweep = {}
    for k in k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sweep[k] = float(silhouette_score(Z, km.labels_))
    k = max(sweep, key=sweep.get)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z).labels_)
            for s in STABILITY_SEEDS]
    return ClusterModel(scaler, pca, km, k, sweep[k], float(np.mean(aris)), km.labels_, small, sweep)


def assign(model: ClusterModel, X: np.ndarray) -> np.ndarray:
    return model.kmeans.predict(_project(model.scaler, model.pca, X))


def summarize(model: ClusterModel, rows, labels: np.ndarray, zone_names: list[str],
              n_bins: int = 5, fit_mask: np.ndarray | None = None) -> list[dict]:
    """Per-cluster summary. `fit_mask` marks which `rows` shaped the centroids (all of them when
    `None`); `n` stays the total membership while `n_fit` counts only the fit ones, and `members`
    (used to pick montage tiles) lists fit members only, nearest first.
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
        occ = np.mean([rows[i].vector[: n_bins * len(zone_names)] for i in idx], axis=0)
        occ = occ.reshape(n_bins, len(zone_names))
        idx_fit = idx[fit_mask[idx]]
        dists = np.linalg.norm(Z[idx_fit] - model.kmeans.cluster_centers_[c], axis=1)
        order = idx_fit[np.argsort(dists)]
        top = [zone_names[j] for j in np.argsort(-occ[1])[:5]]
        path = []
        for b in range(n_bins):
            hs = [int(rows[i].handler[b]) for i in idx if rows[i].handler[b] >= 0]
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
