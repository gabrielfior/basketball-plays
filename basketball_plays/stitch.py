"""Stitch tracker fragments of the same player back together within a possession."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RawTrack:
    track_id: int
    frames: list[int]  # positions within the possession's frame list, increasing
    xy: list[tuple[float, float]]  # court feet, aligned with frames
    boxes: list[list[float]]  # pixel boxes, aligned with frames
    cluster: int | None
    members: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.members:
            self.members = [self.track_id]

    @property
    def start(self) -> int:
        return self.frames[0]

    @property
    def end(self) -> int:
        return self.frames[-1]

    def velocity(self, n: int = 5) -> np.ndarray:
        pts = np.array(self.xy[-n:])
        if len(pts) < 2:
            return np.zeros(2)
        return (pts[-1] - pts[0]) / (len(pts) - 1)


def _compatible(a: RawTrack, b: RawTrack, short_frames: int) -> bool:
    """Team clusters must agree, unless one track is too short for its cluster vote to be trusted."""
    if a.cluster is None or b.cluster is None or a.cluster == b.cluster:
        return True
    return len(a.frames) < short_frames or len(b.frames) < short_frames


def stitch_tracks(
    tracks: list[RawTrack],
    fps: float,
    max_gap_s: float = 2.5,
    max_dist_ft: float = 6.0,
    same_frame_dist_ft: float = 3.0,
    short_track_s: float = 2.0,
) -> list[RawTrack]:
    """Greedily link a track that ends to the nearest track that starts shortly after it.

    A link needs: a gap of 0 to `max_gap_s` seconds (0 = the ids overlap on one frame), the same team cluster (or unknown), and
    the later track starting within `max_dist_ft` of where the earlier one was heading. Each
    track is linked at most once on each side. Linked tracks keep the earliest id.
    """
    tracks = sorted(tracks, key=lambda t: t.start)
    max_gap = int(round(max_gap_s * fps))
    short_frames = int(round(short_track_s * fps))
    consumed: set[int] = set()
    by_id = {t.track_id: t for t in tracks}
    order = [t.track_id for t in tracks]

    for tid in order:
        if tid in consumed:
            continue
        head = by_id[tid]
        while True:
            best, best_d = None, None
            for cand in tracks:
                if cand.track_id in consumed or cand.track_id == head.track_id:
                    continue
                gap = cand.start - head.end
                if gap < 0 or gap > max_gap or not _compatible(head, cand, short_frames):
                    continue
                predicted = np.array(head.xy[-1]) + head.velocity() * gap
                d = float(np.linalg.norm(np.array(cand.xy[0]) - predicted))
                # a same-frame handoff (duplicate box at the id switch) must be very close
                limit = min(max_dist_ft, same_frame_dist_ft) if gap == 0 else max_dist_ft
                if d <= limit and (best_d is None or d < best_d):
                    best, best_d = cand, d
            if best is None:
                break
            skip = 1 if best.start == head.end else 0
            head.frames += best.frames[skip:]
            head.xy += best.xy[skip:]
            head.boxes += best.boxes[skip:]
            head.members += best.members
            if head.cluster is None:
                head.cluster = best.cluster
            consumed.add(best.track_id)
    return [t for t in tracks if t.track_id not in consumed]
