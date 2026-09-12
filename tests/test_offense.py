from basketball_plays import possessions as P


def test_segment_offense_uses_its_own_half_time_mapping_not_the_whole_game_average():
    # First half (video t in [0, 1200s]): cluster 1 attacks on the RIGHT, cluster 0 on the LEFT.
    # After half time (t in [2400, 3600s]) the teams swap baskets: cluster 0 now attacks RIGHT,
    # cluster 1 attacks LEFT. A whole-game learn_offense_map sees the RIGHT-half votes split
    # evenly between cluster 1 (first half) and cluster 0 (second half) and would get half of
    # these segments wrong; segment_offense must decide each segment from its own votes instead.
    seg_specs = [
        (P.LEFT, 0, 0),
        (P.RIGHT, 1, 400),
        (P.LEFT, 0, 800),
        (P.RIGHT, 1, 1200),
        (P.LEFT, 1, 2400),
        (P.RIGHT, 0, 2800),
        (P.LEFT, 1, 3200),
        (P.RIGHT, 0, 3600),
    ]
    states = []
    votes_by_frame = {}
    segments = []
    pos = 0
    for half, cluster, t in seg_specs:
        idx = list(range(pos, pos + 20))
        for i in idx:
            states.append(P.FrameState(frame_idx=i, t=t, valid=True, action_half=half))
            votes_by_frame[i] = [cluster]
        segments.append(P.Segment(frame_indices=idx, half=half))
        pos += 20

    results = [P.segment_offense(votes_by_frame, seg, states, segments) for seg in segments]
    assert results == [0, 1, 0, 1, 1, 0, 1, 0]


def test_segment_with_too_few_votes_falls_back_to_nearby_segments():
    states = []
    votes_by_frame = {}
    segments = []
    pos = 0

    def add_segment(half, cluster, count, t):
        nonlocal pos
        idx = list(range(pos, pos + count))
        for i in idx:
            states.append(P.FrameState(frame_idx=i, t=t, valid=True, action_half=half))
            votes_by_frame[i] = [cluster]
        segments.append(P.Segment(frame_indices=idx, half=half))
        pos += count
        return segments[-1]

    add_segment(P.RIGHT, 1, 20, 0.0)  # neighbour: RIGHT attacked by cluster 1
    seg_target = add_segment(P.RIGHT, 0, 2, 100.0)  # only 2 votes, and they disagree
    add_segment(P.LEFT, 0, 20, 200.0)  # neighbour: LEFT attacked by cluster 0

    result = P.segment_offense(votes_by_frame, seg_target, states, segments, window_s=300.0)
    assert result == 1  # follows the neighbourhood mapping (RIGHT -> 1), not its own 2 votes


def test_segment_with_no_votes_and_no_nearby_segments_uses_global_map_then_none():
    idx = list(range(5))
    states = [P.FrameState(frame_idx=i, t=10_000.0, valid=True, action_half=P.RIGHT) for i in idx]
    seg = P.Segment(frame_indices=idx, half=P.RIGHT)
    votes_by_frame: dict[int, list[int]] = {}

    global_map = {P.LEFT: 0, P.RIGHT: 1}
    assert P.segment_offense(votes_by_frame, seg, states, [seg], global_map=global_map) == 1
    assert P.segment_offense(votes_by_frame, seg, states, [seg]) is None
