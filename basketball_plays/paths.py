"""Trajectory cleaning: teleport removal, gap interpolation, smoothing.

Adapted from roboflow/sports `clean_paths` (MIT), reduced to a single (T, 2) path and using a
moving average so scipy is not required.
"""

from __future__ import annotations

import numpy as np


def interpolate_nan(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    nan = np.isnan(y)
    if nan.all() or not nan.any():
        return y
    idx = np.arange(len(y))
    y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    return y


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    k = min(window, n if n % 2 == 1 else n - 1)
    if k < 3:
        return y
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, np.ones(k) / k, mode="valid")


def clean_path(
    xy: np.ndarray,
    jump_sigma: float = 5.0,
    min_jump_dist: float = 3.0,
    max_jump_run: int = 6,
    pad_around_runs: int = 1,
    smooth_window: int = 5,
) -> np.ndarray:
    """Clean one (T, 2) court-space path; NaNs mark missing frames.

    Frame-to-frame jumps that are both far larger than the path's typical speed and above
    `min_jump_dist` feet are treated as teleports when they last at most `max_jump_run`
    frames, removed, and linearly re-interpolated with the other gaps; then a moving-average
    smooth is applied per axis.
    """
    xy = np.asarray(xy, dtype=float).copy()
    if xy.ndim != 2 or xy.shape[0] == 0:
        return xy
    if np.isnan(xy).all():
        return xy
    T = xy.shape[0]
    valid = ~np.isnan(xy).any(axis=1)
    remove = np.zeros(T, dtype=bool)
    if valid.sum() >= 3:
        filled = np.stack([interpolate_nan(xy[:, 0]), interpolate_nan(xy[:, 1])], axis=1)
        speed = np.linalg.norm(np.diff(filled, axis=0), axis=1)
        med = np.median(speed)
        mad = 1.4826 * np.median(np.abs(speed - med))
        scale = max(mad, 1e-6)
        jump = (speed > med + jump_sigma * scale) & (speed > min_jump_dist)
        # a teleport is a jump out followed within max_jump_run frames by a jump back
        idx = np.flatnonzero(jump)
        i = 0
        while i < len(idx):
            start = idx[i]
            j = i + 1
            while j < len(idx) and idx[j] - start <= max_jump_run:
                j += 1
            if j - i >= 2:  # out and back
                end = idx[j - 1]
                rs = max(0, start + 1 - pad_around_runs)
                re = min(T - 1, end + pad_around_runs)
                remove[rs : re + 1] = True
            i = max(j, i + 1)
    # never throw away most of a track: if the "teleports" are the whole track, keep it as is
    if remove.any() and (valid & ~remove).sum() < max(2, int(0.5 * valid.sum())):
        remove[:] = False
    xy[remove] = np.nan
    out = np.stack([interpolate_nan(xy[:, 0]), interpolate_nan(xy[:, 1])], axis=1)
    for d in range(2):
        out[:, d] = _smooth(out[:, d], smooth_window)
    return out
