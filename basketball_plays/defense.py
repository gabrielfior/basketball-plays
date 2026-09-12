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
FOLLOW_DT = 1.0  # displacement window for the "follow" feature
FOLLOW_TOL = FOLLOW_DT * 0.2  # tolerance when hunting for the frame nearest FOLLOW_DT later


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


def _nearest_frame_time(times: list[float], target: float, tol: float) -> float | None:
    """The time in `times` closest to `target`, or `None` when none is within `tol`."""
    best = None
    for t in times:
        d = abs(t - target)
        if d <= tol and (best is None or d < abs(best - target)):
            best = t
    return best


def features(rec: H.HalfcourtRecord, window_s: float = 8.0) -> dict | None:
    """Matchup-based features over the `window_s` seconds after the setup (or `t0` when there is
    no setup frame). `None` when fewer than `MIN_FRAMES` frames have a usable matchup.

    Per-frame points are read from the Track objects with `track_id` as identity (via
    `halfcourt.id_positions_at`), not from `halfcourt.positions_at`'s own list order:
    `positions_at` is id-agnostic and its ranking (see `halfcourt._rank`) can reorder frame to
    frame on real tracking (detection status and effective track length vary per frame), so
    treating its output index as a stable player identity would measure index churn rather than
    actual defenders. `matchups` itself stays a pure position-based helper (it only needs
    court distance); this function is what turns its index pairs back into `(off_track_id,
    def_track_id)` pairs before using them across frames. Track fragmentation -- a defender's
    tracked id changing mid-possession -- still causes some genuine churn in `stability`/`follow`;
    that is an accepted limitation of the underlying tracking, not something this function can
    recover from.
    """
    off_tracks = H.tracks_from_record(rec)
    def_tracks = H.tracks_from_players(rec.opponents)
    a = rec.setup if rec.setup is not None else rec.t0
    off_times = {t for tr in off_tracks for t in tr.xy}
    def_times = {t for tr in def_tracks for t in tr.xy}
    hi = min(a + window_s, rec.t_end)
    times = sorted(t for t in off_times & def_times if a <= t <= hi)

    # each frame: (t, {off_track_id: xy}, {def_track_id: xy}, {(off_track_id, def_track_id)})
    frames: list[tuple[float, dict[int, tuple[float, float]], dict[int, tuple[float, float]],
                       set[tuple[int, int]]]] = []
    for t in times:
        o_ids = H.id_positions_at(off_tracks, t)
        d_ids = H.id_positions_at(def_tracks, t)
        o_arr = np.array([p for _, p in o_ids]).reshape(-1, 2)
        d_arr = np.array([p for _, p in d_ids]).reshape(-1, 2)
        m = matchups(o_arr, d_arr)
        if not m:
            continue
        o_map = dict(o_ids)
        d_map = dict(d_ids)
        pairs = {(o_ids[ai][0], d_ids[di][0]) for ai, di in m}
        frames.append((t, o_map, d_map, pairs))
    if len(frames) < MIN_FRAMES:
        return None

    # stability: fraction of consecutive frame pairs with an identical (off_id, def_id) set
    stable = sum(1 for (_, _, _, p1), (_, _, _, p2) in pairwise(frames) if p1 == p2)
    stability = stable / (len(frames) - 1)

    # follow: mean Pearson-style correlation of defender and matched-attacker displacement,
    # each pair followed by track_id to the frame nearest FOLLOW_DT seconds later
    frame_times = [f[0] for f in frames]
    by_time = {f[0]: f for f in frames}
    follows = []
    for t0, o0, d0, pairs0 in frames:
        t1 = _nearest_frame_time(frame_times, t0 + FOLLOW_DT, FOLLOW_TOL)
        if t1 is None:
            continue
        _, o1, d1, _ = by_time[t1]
        for off_id, def_id in pairs0:
            if off_id in o1 and def_id in d1:
                do = np.array(o1[off_id]) - np.array(o0[off_id])
                dd = np.array(d1[def_id]) - np.array(d0[def_id])
                if np.linalg.norm(do) > 0.5 and np.linalg.norm(dd) > 0.5:
                    follows.append(float(np.dot(do, dd) / (np.linalg.norm(do) * np.linalg.norm(dd))))
    follow = float(np.mean(follows)) if follows else 0.0

    # spread: mean, over defenders (grouped by def_track_id), of the std of their position
    # over the window
    per_def: dict[int, list[tuple[float, float]]] = {}
    for _, _, d_map, pairs in frames:
        for _, def_id in pairs:
            per_def.setdefault(def_id, []).append(d_map[def_id])
    spread = float(np.mean([np.std(np.array(v), axis=0).mean() for v in per_def.values()]))

    gap = float(np.median([np.linalg.norm(np.array(o_map[off_id]) - np.array(d_map[def_id]))
                            for _, o_map, d_map, pairs in frames for off_id, def_id in pairs]))
    paint = float(np.mean([np.sum(Z.rim_distance(np.array(list(d_map.values()))) < PAINT_FT)
                           for _, _, d_map, _ in frames]))
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
