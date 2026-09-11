"""Canonical attack direction and a polar zone grid around the attacking rim.

Duke always attacks the basket at x = 5.25 after mirroring. Zones are four distance rings split
into five 36-degree sectors from the far sideline (small y) to the near sideline (large y), plus
`deep` (32 ft or more, frontcourt) and `backcourt` (x >= 47). Zone id = ring * 5 + sector.
"""

from __future__ import annotations

import math

import numpy as np

from basketball_plays.court import NCAA

RIM = (NCAA.rim_offset, NCAA.width / 2)  # (5.25, 25.0)
HALF_COURT_X = NCAA.length / 2  # 47.0
RINGS = (8.0, 16.0, NCAA.three_point_radius, 32.0)  # outer edge of rings 0..3
SECTOR_DEG = 36.0
N_SECTORS = 5
DEEP = 20
BACKCOURT = 21
N_ZONES = 22

ZONE_NAMES: list[str] = [
    "dunker_far", "rim_far", "rim", "rim_near", "dunker_near",            # ring 0: 0-8 ft
    "block_far", "post_far", "paint", "post_near", "block_near",          # ring 1: 8-16 ft
    "baseline_mid_far", "elbow_far", "nail", "elbow_near", "baseline_mid_near",  # ring 2: 16-22.15
    "corner_far", "wing_far", "top", "wing_near", "corner_near",          # ring 3: 22.15-32
    "deep", "backcourt",
]
assert len(ZONE_NAMES) == N_ZONES


def mirror_to_canonical(xy: np.ndarray, attacking_basket: str) -> np.ndarray:
    """Copy of (n, 2) court points with x flipped when the offence attacks the right basket.

    Raises `ValueError` on anything but "left" or "right": a typo or a None would otherwise pass
    silently as "no mirroring" and leave half the data 180 degrees out.
    """
    if attacking_basket not in ("left", "right"):
        raise ValueError(f"attacking_basket must be 'left' or 'right', got {attacking_basket!r}")
    out = np.array(xy, dtype=float, copy=True).reshape(-1, 2)
    if attacking_basket == "right":
        out[:, 0] = NCAA.length - out[:, 0]
    return out


def rim_distance(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.hypot(xy[:, 0] - RIM[0], xy[:, 1] - RIM[1])


def zone_of(x: float, y: float) -> int:
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("zone_of needs finite coordinates")
    if x >= HALF_COURT_X:
        return BACKCOURT
    dx, dy = x - RIM[0], y - RIM[1]
    dist = math.hypot(dx, dy)
    ring = next((i for i, edge in enumerate(RINGS) if dist < edge), None)
    if ring is None:
        return DEEP
    # angle 0 points up court; -90 is the far sideline, +90 the near sideline; points behind the
    # rim line fold onto the nearest baseline sector
    angle = math.degrees(math.atan2(dy, dx))
    angle = max(-90.0, min(90.0, angle))
    sector = min(N_SECTORS - 1, int((angle + 90.0) // SECTOR_DEG))
    return ring * N_SECTORS + sector


def zones_of(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    return np.array([zone_of(float(x), float(y)) for x, y in xy], dtype=int)


def occupancy(xy: np.ndarray) -> np.ndarray:
    """Count of points per zone, NaN rows ignored."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    out = np.zeros(N_ZONES, dtype=float)
    if len(xy):
        np.add.at(out, zones_of(xy), 1.0)
    return out
