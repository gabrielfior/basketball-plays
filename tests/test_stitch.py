import numpy as np

from basketball_plays import stitch


def track(tid, frames, xs, y=25.0, cluster=0):
    return stitch.RawTrack(track_id=tid, frames=list(frames), xy=[(x, y) for x in xs],
                           boxes=[[0, 0, 10, 10]] * len(frames), cluster=cluster)


def test_stitch_joins_consecutive_fragments_of_one_player():
    a = track(1, range(20), np.linspace(10, 20, 20))
    b = track(2, range(22, 40), np.linspace(21, 30, 18))  # starts 0.2 s after a ends, 1 ft away
    merged = stitch.stitch_tracks([a, b], fps=10, max_gap_s=1.5, max_dist_ft=6.0)
    assert len(merged) == 1
    assert merged[0].track_id == 1
    assert merged[0].frames == list(range(20)) + list(range(22, 40))
    assert merged[0].members == [1, 2]


def test_stitch_respects_team_and_distance():
    a = track(1, range(20), np.linspace(10, 20, 20), cluster=0)
    far = track(2, range(22, 40), np.linspace(60, 70, 18), cluster=0)
    other_team = track(3, range(22, 40), np.linspace(21, 30, 18), cluster=1)
    merged = stitch.stitch_tracks([a, far, other_team], fps=10)
    assert sorted(m.track_id for m in merged) == [1, 2, 3]


def test_stitch_does_not_join_overlapping_tracks():
    a = track(1, range(20), np.linspace(10, 20, 20))
    b = track(2, range(15, 40), np.linspace(18, 30, 25))
    merged = stitch.stitch_tracks([a, b], fps=10)
    assert len(merged) == 2


def test_stitch_picks_nearest_candidate_and_chains():
    a = track(1, range(10), np.linspace(10, 12, 10))
    near = track(2, range(12, 20), np.linspace(12.5, 14, 8))
    farther = track(3, range(12, 20), np.linspace(16, 18, 8))
    tail = track(4, range(22, 30), np.linspace(14.5, 16, 8))
    merged = stitch.stitch_tracks([a, near, farther, tail], fps=10)
    ids = {m.track_id: m for m in merged}
    assert set(ids) == {1, 3}
    assert ids[1].members == [1, 2, 4]


def test_unknown_cluster_can_join_either_team():
    a = track(1, range(10), np.linspace(10, 12, 10), cluster=1)
    b = track(2, range(12, 20), np.linspace(12.5, 14, 8), cluster=None)
    merged = stitch.stitch_tracks([a, b], fps=10)
    assert len(merged) == 1 and merged[0].cluster == 1


def test_same_frame_handoff_is_joined_when_close_and_frames_stay_unique():
    a = track(1, range(10), np.linspace(10, 12, 10))
    b = track(2, range(9, 20), np.linspace(12.5, 15, 11))  # overlaps a on frame 9, 0.5 ft away
    merged = stitch.stitch_tracks([a, b], fps=10)
    assert len(merged) == 1
    assert merged[0].frames == list(range(20))
    far = track(3, range(9, 20), np.linspace(16, 18, 11))  # 4 ft away: too far for a same-frame handoff
    merged = stitch.stitch_tracks([track(1, range(10), np.linspace(10, 12, 10)), far], fps=10)
    assert len(merged) == 2
