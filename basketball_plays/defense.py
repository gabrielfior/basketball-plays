"""Defensive scheme from matchups: Hungarian assignment per frame, five features over the eight
seconds after the setup, and a two-cluster rule labelling man versus zone (spec section 5b)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from basketball_plays import halfcourt as H
from basketball_plays import zones as Z

MIN_PLAYERS = 4
MIN_FRAMES = 20
PAINT_FT = 16.0
UNKNOWN_MARGIN = 0.25
FEATURE_KEYS = ("stability", "follow", "spread", "gap", "paint")


def matchups(off: np.ndarray, deff: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one attacker-to-defender assignment minimising total court distance.

    Requires at least `MIN_PLAYERS` of each side; otherwise the frame is unusable and `[]` is
    returned rather than a partial assignment.
    """
    if len(off) < MIN_PLAYERS or len(deff) < MIN_PLAYERS:
        return []
    cost = np.linalg.norm(off[:, None, :] - deff[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def _positions(tracks: list[H.Track], t: float) -> np.ndarray:
    return np.array(H.positions_at(tracks, t)).reshape(-1, 2)


def features(rec: H.HalfcourtRecord, window_s: float = 8.0) -> dict | None:
    """Matchup-based features over the `window_s` seconds after the setup (or `t0` when there is
    no setup frame). `None` when fewer than `MIN_FRAMES` frames have a usable matchup."""
    off_tracks = H.tracks_from_record(rec)
    def_tracks = H.tracks_from_players(rec.opponents)
    a = rec.setup if rec.setup is not None else rec.t0
    off_times = {t for tr in off_tracks for t in tr.xy}
    def_times = {t for tr in def_tracks for t in tr.xy}
    hi = min(a + window_s, rec.t_end)
    times = sorted(t for t in off_times & def_times if a <= t <= hi)
    frames = []
    for t in times:
        o, d = _positions(off_tracks, t), _positions(def_tracks, t)
        m = matchups(o, d)
        if m:
            frames.append((t, o, d, m))
    if len(frames) < MIN_FRAMES:
        return None

    # stability: fraction of consecutive frame pairs with an identical matchup set
    stable = sum(1 for (_, _, _, m1), (_, _, _, m2) in pairwise(frames)
                 if set(m1) == set(m2))
    stability = stable / (len(frames) - 1)

    # follow: mean Pearson-style correlation of defender and matched-attacker displacement,
    # over roughly 1 second (10 frames at the 10 fps the tracks were sampled at)
    follows = []
    for i in range(len(frames) - 10):
        _, o0, d0, m0 = frames[i]
        _, o1, d1, _ = frames[i + 10]
        for ai, di in m0:
            if ai < len(o1) and di < len(d1):
                do, dd = o1[ai] - o0[ai], d1[di] - d0[di]
                if np.linalg.norm(do) > 0.5 and np.linalg.norm(dd) > 0.5:
                    follows.append(float(np.dot(do, dd) / (np.linalg.norm(do) * np.linalg.norm(dd))))
    follow = float(np.mean(follows)) if follows else 0.0

    # spread: mean, over defenders (grouped by matched attacker slot), of the std of their
    # position over the window
    per_slot: dict[int, list[np.ndarray]] = {}
    for _, _, d, m in frames:
        for ai, di in m:
            per_slot.setdefault(ai, []).append(d[di])
    spread = float(np.mean([np.std(np.array(v), axis=0).mean() for v in per_slot.values()]))

    gap = float(np.median([np.linalg.norm(o[ai] - d[di])
                            for _, o, d, m in frames for ai, di in m]))
    paint = float(np.mean([np.sum(Z.rim_distance(d) < PAINT_FT) for _, _, d, _ in frames]))
    return {"stability": stability, "follow": follow, "spread": spread, "gap": gap,
            "paint": paint, "frames": len(frames)}


def label_rule(feats: list[dict | None], seed: int = 0) -> list[tuple[str, float]]:
    """Standardise the feature vectors, k-means with k=2, and label the cluster with the higher
    mean (stability + follow) `man` and the other `zone`. Confidence is the margin between
    distance to each centroid, normalised to [0, 1] by the inter-centroid distance; below
    `UNKNOWN_MARGIN` (or a `None` input) the record is labelled `unknown` with confidence 0.
    """
    idx = [i for i, f in enumerate(feats) if f is not None]
    out: list[tuple[str, float]] = [("unknown", 0.0)] * len(feats)
    if len(idx) < 4:
        return out
    X = np.array([[feats[i][k] for k in FEATURE_KEYS] for i in idx])
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(Xs)
    score = [np.mean(X[km.labels_ == c][:, :2]) for c in (0, 1)]  # stability + follow
    man_c = int(np.argmax(score))
    d = np.linalg.norm(Xs[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    centroid_dist = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])
    margin = np.abs(d[:, 0] - d[:, 1]) / (centroid_dist + 1e-9)
    for j, i in enumerate(idx):
        conf = float(min(1.0, margin[j]))
        lab = ("man" if km.labels_[j] == man_c else "zone") if conf >= UNKNOWN_MARGIN else "unknown"
        out[i] = (lab, conf)
    return out
