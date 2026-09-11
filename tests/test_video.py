import numpy as np

from basketball_plays import video


def test_sample_indices_at_target_fps():
    idx = video.sample_indices(src_fps=60.0, target_fps=10.0, start_s=0.0, end_s=1.0)
    assert idx == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_sample_indices_offsets_and_empty():
    assert video.sample_indices(30.0, 10.0, 2.0, 2.3) == [60, 63, 66]
    assert video.sample_indices(30.0, 10.0, 5.0, 5.0) == []


def test_shot_change_detects_hard_cut():
    a = np.zeros((90, 160, 3), dtype=np.uint8)
    a[:, :, 2] = 200  # red-ish
    b = np.zeros((90, 160, 3), dtype=np.uint8)
    b[:, :, 0] = 200  # blue-ish
    _, h = video.shot_changed(None, a)
    changed, _ = video.shot_changed(h, a.copy())
    assert changed is False
    changed, _ = video.shot_changed(h, b)
    assert changed is True
