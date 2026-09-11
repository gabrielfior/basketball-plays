import numpy as np
import pytest

from basketball_plays import court, geometry


def synthetic_camera():
    """A plausible broadcast-like homography from court feet to 1280x720 pixels."""
    src = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float32)
    dst = np.array([[200, 150], [1080, 150], [1250, 700], [30, 700]], dtype=np.float32)
    import cv2

    H, _ = cv2.findHomography(src, dst)
    return H


def project(H, pts):
    import cv2

    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def test_fit_homography_recovers_camera():
    H_true = synthetic_camera()
    verts = np.array(court.NCAA.vertices, dtype=np.float32)
    pix = project(H_true, verts)
    conf = np.ones(len(verts))
    keypoints = np.concatenate([pix, conf[:, None]], axis=1)
    fit = geometry.fit_homography(keypoints, min_conf=0.5)
    assert fit is not None
    back = fit.to_court(pix[[0, 16, 32]])
    assert np.allclose(back, verts[[0, 16, 32]], atol=0.05)
    assert fit.reprojection_error < 0.1


def test_fit_homography_uses_only_confident_points_and_needs_four():
    H_true = synthetic_camera()
    verts = np.array(court.NCAA.vertices, dtype=np.float32)
    pix = project(H_true, verts)
    conf = np.zeros(len(verts))
    conf[[0, 5, 27]] = 1.0
    keypoints = np.concatenate([pix, conf[:, None]], axis=1)
    assert geometry.fit_homography(keypoints, min_conf=0.5) is None
    conf[[32]] = 1.0
    keypoints = np.concatenate([pix, conf[:, None]], axis=1)
    assert geometry.fit_homography(keypoints, min_conf=0.5) is not None


def test_fit_homography_rejects_garbage_points():
    rng = np.random.default_rng(0)
    keypoints = np.concatenate(
        [rng.uniform(0, 1280, (33, 1)), rng.uniform(0, 720, (33, 1)), np.ones((33, 1))], axis=1
    )
    fit = geometry.fit_homography(keypoints, min_conf=0.5, max_reprojection_error=2.0)
    assert fit is None


def test_fit_is_robust_to_one_outlier_keypoint():
    H_true = synthetic_camera()
    verts = np.array(court.NCAA.vertices, dtype=np.float32)
    pix = project(H_true, verts)
    pix[10] += np.array([300, -200])  # one badly placed landmark
    keypoints = np.concatenate([pix, np.ones((33, 1))], axis=1)
    fit = geometry.fit_homography(keypoints, min_conf=0.5)
    assert fit is not None
    back = fit.to_court(project(H_true, np.array([[47.0, 25.0]])))
    assert np.allclose(back, [[47.0, 25.0]], atol=0.5)


def test_inside_court_with_margin():
    pts = np.array([[1.0, 1.0], [-3.0, 25.0], [47.0, 55.0], [96.0, 10.0]])
    assert geometry.inside_court(pts, margin=0.0).tolist() == [True, False, False, False]
    assert geometry.inside_court(pts, margin=6.0).tolist() == [True, True, True, True]


def test_bottom_center():
    boxes = np.array([[10, 20, 30, 60], [0, 0, 4, 4]], dtype=float)
    assert geometry.bottom_center(boxes).tolist() == [[20.0, 60.0], [2.0, 4.0]]


@pytest.mark.parametrize("n", [0])
def test_to_court_handles_empty(n):
    H_true = synthetic_camera()
    verts = np.array(court.NCAA.vertices, dtype=np.float32)
    pix = project(H_true, verts)
    fit = geometry.fit_homography(np.concatenate([pix, np.ones((33, 1))], axis=1))
    assert fit.to_court(np.zeros((n, 2))).shape == (0, 2)
