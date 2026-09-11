"""Video sampling helpers shared by the GPU stage and the overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    try:
        return VideoMeta(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def sample_indices(src_fps: float, target_fps: float, start_s: float, end_s: float) -> list[int]:
    """Source frame indices that sample [start_s, end_s) at target_fps."""
    if target_fps <= 0 or src_fps <= 0 or end_s <= start_s:
        return []
    times = np.arange(start_s, end_s, 1.0 / target_fps)
    idx = np.unique(np.round(times * src_fps).astype(int))
    return idx.tolist()


def iter_frames(path: str, indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame) for the given sorted source indices, reading sequentially."""
    if not indices:
        return
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        pos = indices[0]
        wanted = iter(indices)
        target = next(wanted)
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if pos == target:
                yield pos, frame
                try:
                    target = next(wanted)
                except StopIteration:
                    return
            pos += 1
    finally:
        cap.release()


def shot_changed(prev_hist: np.ndarray | None, frame: np.ndarray, threshold: float = 0.5):
    """Detect a hard camera cut with an HSV histogram distance. Returns (changed, hist)."""
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    if prev_hist is None:
        return False, hist
    dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
    return bool(dist > threshold), hist
