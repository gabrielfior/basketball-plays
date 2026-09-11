import numpy as np

from basketball_plays import paths


def test_clean_path_removes_teleport_and_fills_gap():
    t = np.arange(60)
    xy = np.stack([t * 0.3, np.full(60, 25.0)], axis=1)
    xy[30] = [80.0, 5.0]  # one-frame teleport
    xy[40:43] = np.nan  # short gap
    cleaned = paths.clean_path(xy)
    assert cleaned.shape == xy.shape
    assert not np.isnan(cleaned).any()
    assert abs(cleaned[30, 0] - 9.0) < 1.0
    assert abs(cleaned[41, 0] - 12.3) < 1.0


def test_clean_path_short_input_is_returned_finite():
    xy = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0]])
    cleaned = paths.clean_path(xy)
    assert cleaned.shape == (3, 2)
    assert not np.isnan(cleaned).any()


def test_clean_path_all_nan_stays_nan():
    xy = np.full((5, 2), np.nan)
    assert np.isnan(paths.clean_path(xy)).all()


def test_interpolate_nan_edges_are_held():
    y = np.array([np.nan, 1.0, np.nan, 3.0, np.nan])
    assert paths.interpolate_nan(y).tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
