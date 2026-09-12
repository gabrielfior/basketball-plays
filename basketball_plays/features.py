"""Fixed-length features for one half-court record: zone occupancy by time bin, ball-handler zone
sequence, identity snapshot at the setup frame, and context. See spec section 4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from basketball_plays import halfcourt as H
from basketball_plays import zones as Z

BINS = ((-1.0, 0.0), (0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, None))
BUCKETS = ("ato", "inbound", "after_score")
NOT_VISIBLE = Z.N_ZONES  # index 22 in the snapshot one-hot


def in_scope(rec: H.HalfcourtRecord) -> bool:
    """Whether a half-court record is one of the dead-ball possessions this pipeline models.

    Every stage -- features, defence, the page -- has to agree on this, or the row sets drift
    apart and the per-record keys stop lining up.
    """
    return (rec.start_type in ("ato", "dead") and rec.located and not rec.transition
            and rec.t0 is not None)


def bucket(rec: H.HalfcourtRecord) -> str:
    if rec.start_type == "ato":
        return "ato"
    return "after_score" if rec.full_court else "inbound"


def _anchor(rec: H.HalfcourtRecord) -> float:
    return rec.setup if rec.setup is not None else rec.t0


def _bin_range(rec: H.HalfcourtRecord, b: int) -> tuple[float, float]:
    lo, hi = BINS[b]
    a = _anchor(rec)
    return a + lo, (rec.t_end if hi is None else a + hi)


def occupancy_by_bin(rec: H.HalfcourtRecord) -> np.ndarray:
    tracks = H.tracks_from_record(rec)
    times = sorted({t for tr in tracks for t in tr.xy})
    out = np.zeros((len(BINS), Z.N_ZONES))
    for b in range(len(BINS)):
        lo, hi = _bin_range(rec, b)
        frames = [t for t in times if lo <= t < hi]
        if not frames:
            continue
        acc = np.zeros(Z.N_ZONES)
        for t in frames:
            acc += Z.occupancy(np.array(H.positions_at(tracks, t)).reshape(-1, 2))
        out[b] = acc / len(frames)
    return out


def handler_zones(rec: H.HalfcourtRecord) -> np.ndarray:
    out = np.full(len(BINS), -1, dtype=int)
    for b in range(len(BINS)):
        lo, hi = _bin_range(rec, b)
        zs = [Z.zone_of(x, y) for t, x, y in rec.ball_handler if lo <= t < hi]
        if zs:
            out[b] = int(np.bincount(zs).argmax())
    return out


def identity_snapshot(rec: H.HalfcourtRecord, roster: list[str]) -> np.ndarray:
    snap = np.zeros((len(roster), Z.N_ZONES + 1))
    a = _anchor(rec)
    for i, name in enumerate(roster):
        pos = None
        for p in rec.players:
            if p.get("name") != name:
                continue
            for row in p["trajectory"]:
                if abs(row[0] - a) < 0.051 and np.isfinite(row[1]):
                    pos = (row[1], row[2])
                    break
            if pos:
                break
        snap[i, Z.zone_of(*pos) if pos else NOT_VISIBLE] = 1.0
    return snap


def context(rec: H.HalfcourtRecord) -> dict:
    clock = rec.clock_start
    clock_bucket = "late" if clock <= 120 else ("mid" if clock <= 600 else "early")
    margin_bucket = "unknown"
    sb = getattr(rec, "score_before", None)
    if isinstance(sb, dict) and len(sb) == 2 and rec.team in sb:
        us = sb[rec.team]
        them = next(v for k, v in sb.items() if k != rec.team)
        d = us - them
        margin_bucket = "down" if d < -5 else ("up" if d > 5 else "close")
    return {"bucket": bucket(rec), "no_setup": rec.no_setup, "period": rec.period,
            "clock_bucket": clock_bucket, "margin_bucket": margin_bucket}


def cluster_vector(rec: H.HalfcourtRecord) -> np.ndarray:
    occ = occupancy_by_bin(rec).ravel()
    hz = handler_zones(rec).astype(float)
    hz = np.where(hz < 0, -1.0, hz / Z.N_ZONES)
    onehot = np.array([1.0 if bucket(rec) == b else 0.0 for b in BUCKETS])
    return np.concatenate([occ, hz, onehot])


@dataclass
class FeatureRow:
    game_id: str
    period: int
    index: int
    bucket: str
    split: str
    no_setup: bool
    vector: np.ndarray
    handler: np.ndarray
    snapshot: np.ndarray | None = None
    context: dict = field(default_factory=dict)


def build_rows(records: list[H.HalfcourtRecord], splits_by_game: dict[str, str],
               roster: list[str] | None = None) -> list[FeatureRow]:
    rows = []
    for r in records:
        rows.append(FeatureRow(
            game_id=r.game_id, period=r.period, index=r.index, bucket=bucket(r),
            split=splits_by_game.get(r.game_id, "train"), no_setup=r.no_setup,
            vector=cluster_vector(r), handler=handler_zones(r),
            snapshot=identity_snapshot(r, roster) if roster else None, context=context(r),
        ))
    return rows
