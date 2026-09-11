"""Frame-to-court homography fitting and point projection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from basketball_plays.court import NCAA, CourtSpec


@dataclass
class HomographyFit:
    matrix: np.ndarray  # 3x3, pixel -> court feet
    n_points: int
    reprojection_error: float  # mean court-space error (ft) over inlier landmarks

    def to_court(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts = pixels.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def to_pixels(self, court_xy: np.ndarray) -> np.ndarray:
        court_xy = np.asarray(court_xy, dtype=np.float32)
        if court_xy.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        inv = np.linalg.inv(self.matrix)
        return cv2.perspectiveTransform(court_xy.reshape(-1, 1, 2), inv).reshape(-1, 2)

    def to_list(self) -> list[list[float]]:
        return self.matrix.tolist()


def fit_homography(
    keypoints: np.ndarray,
    min_conf: float = 0.5,
    max_reprojection_error: float = 2.0,
    min_inlier_fraction: float = 0.6,
    spec: CourtSpec = NCAA,
) -> HomographyFit | None:
    """Fit pixel->court homography from (33, 3) keypoints [x, y, conf].

    Uses RANSAC so a single misplaced landmark does not ruin the fit. Returns None when fewer
    than four confident landmarks exist or the inlier reprojection error is too large.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] == 0:
        return None
    verts = spec.vertices_array()
    n = min(len(verts), keypoints.shape[0])
    mask = keypoints[:n, 2] > min_conf
    if mask.sum() < 4:
        return None
    src = keypoints[:n][mask, :2]
    dst = verts[:n][mask]
    if mask.sum() == 4:
        H, inliers = cv2.findHomography(src, dst, 0)
        inliers = np.ones((4, 1), dtype=np.uint8) if H is not None else None
    else:
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None or inliers is None:
        return None
    inl = inliers.ravel().astype(bool)
    # a real court view has most confident landmarks agreeing with one homography
    if inl.sum() < max(4, int(np.ceil(min_inlier_fraction * mask.sum()))):
        return None
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(proj - dst[inl], axis=1)))
    if not np.isfinite(err) or err > max_reprojection_error:
        return None
    return HomographyFit(matrix=H, n_points=int(inl.sum()), reprojection_error=err)


def inside_court(court_xy: np.ndarray, margin: float = 0.0, spec: CourtSpec = NCAA) -> np.ndarray:
    xy = np.asarray(court_xy)
    if xy.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (xy[:, 0] >= -margin)
        & (xy[:, 0] <= spec.length + margin)
        & (xy[:, 1] >= -margin)
        & (xy[:, 1] <= spec.width + margin)
    )


def bottom_center(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros((0, 2))
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def box_center(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros((0, 2))
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
