import numpy as np

from basketball_plays import possessions as P


def frames(halves, fps=10, start=0):
    """Build a per-frame table: list of (t, valid, half) tuples. half in {-1: left, 1: right, 0: invalid}."""
    out = []
    for i, h in enumerate(halves):
        out.append(P.FrameState(frame_idx=start + i, t=(start + i) / fps, valid=h != 0,
                                action_half=h if h != 0 else None))
    return out


def test_action_half_from_player_positions():
    xy = np.array([[10.0, 5], [20, 25], [30, 40], [15, 10], [8, 30], [60, 25]])
    assert P.action_half(xy) == -1
    assert P.action_half(94 - xy) == 1
    assert P.action_half(np.zeros((0, 2))) is None


def test_single_stable_half_is_one_possession():
    fs = frames([-1] * 60)
    segs = P.segment(fs, fps=10, min_duration=3.0)
    assert len(segs) == 1
    assert segs[0].frame_indices[0] == 0 and segs[0].frame_indices[-1] == 59
    assert segs[0].half == -1


def test_half_flip_splits_possessions():
    fs = frames([-1] * 60 + [1] * 60)
    segs = P.segment(fs, fps=10, min_duration=3.0)
    assert [s.half for s in segs] == [-1, 1]
    assert segs[0].frame_indices[-1] == 59
    assert segs[1].frame_indices[0] == 60


def test_short_flicker_does_not_split():
    fs = frames([-1] * 60 + [1] * 5 + [-1] * 60)
    segs = P.segment(fs, fps=10, min_duration=3.0, min_flip_duration=1.5)
    assert len(segs) == 1
    assert len(segs[0].frame_indices) == 125


def test_short_invalid_gap_is_bridged_long_gap_splits():
    fs = frames([-1] * 60 + [0] * 10 + [-1] * 60)
    assert len(P.segment(fs, fps=10, min_duration=3.0, max_gap=3.0)) == 1
    fs = frames([-1] * 60 + [0] * 50 + [-1] * 60)
    segs = P.segment(fs, fps=10, min_duration=3.0, max_gap=3.0, merge_same_half_gap=0)
    assert len(segs) == 2
    # invalid frames are not part of either possession
    assert all(fs[i].valid for s in segs for i in s.frame_indices)


def test_too_short_runs_are_dropped():
    fs = frames([-1] * 20 + [0] * 50 + [1] * 60)
    segs = P.segment(fs, fps=10, min_duration=3.0)
    assert [s.half for s in segs] == [1]


def test_same_half_runs_across_a_replay_gap_are_merged():
    fs = frames([-1] * 60 + [0] * 80 + [-1] * 60)  # 8 s of replay in the middle
    segs = P.segment(fs, fps=10, min_duration=3.0, max_gap=3.0, merge_same_half_gap=12.0)
    assert len(segs) == 1 and len(segs[0].frame_indices) == 120
    fs = frames([-1] * 60 + [0] * 150 + [-1] * 60)  # 15 s gap: too long to trust
    assert len(P.segment(fs, fps=10, min_duration=3.0, max_gap=3.0, merge_same_half_gap=12.0)) == 2
    fs = frames([-1] * 60 + [0] * 80 + [1] * 60)  # different half: never merged
    assert len(P.segment(fs, fps=10, min_duration=3.0, merge_same_half_gap=12.0)) == 2


def test_offense_map_from_possession_detections():
    # right-half action mostly has cluster 1 in possession; left-half has cluster 0
    votes = [(1, 1), (1, 1), (1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 1)]
    m = P.learn_offense_map(votes)
    assert m == {1: 1, -1: 0}


def test_offense_map_is_consistent_when_one_side_is_missing():
    votes = [(1, 1), (1, 1)]
    assert P.learn_offense_map(votes) == {1: 1, -1: 0}
