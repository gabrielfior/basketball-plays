import numpy as np

from basketball_plays import court, extract
from basketball_plays import possessions as P
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import CLS_PLAYER, CLS_PLAYER_IN_POSSESSION, Detection, FrameRecord
from tests.test_extract import camera, to_pix


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


def test_learn_offense_maps_by_span_learns_each_span_independently():
    # First period [0, 1200s): RIGHT attacked by cluster 1, LEFT by cluster 0. Second period
    # [2400, 3600s): the teams have swapped baskets, so the mapping is the exact inverse.
    votes = []
    for t in range(20):
        votes.append((float(t), P.RIGHT, 1))
        votes.append((float(t), P.LEFT, 0))
    for t in range(2400, 2420):
        votes.append((float(t), P.RIGHT, 0))
        votes.append((float(t), P.LEFT, 1))
    spans = [(0.0, 1200.0), (2400.0, 3600.0)]

    maps = P.learn_offense_maps_by_span(votes, spans)

    assert maps[0] == {P.RIGHT: 1, P.LEFT: 0}
    assert maps[1] == {P.RIGHT: 0, P.LEFT: 1}
    assert maps[0] == {k: 1 - v for k, v in maps[1].items()}

    # a segment's midpoint routes it to its own period's map, not the other period's.
    idx_first_period = P.span_index(500.0, spans)
    idx_second_period = P.span_index(3000.0, spans)
    assert maps[idx_first_period].get(P.RIGHT) == 1
    assert maps[idx_second_period].get(P.RIGHT) == 0


def test_span_index_inside_near_and_far():
    spans = [(0.0, 1200.0), (2400.0, 3600.0)]
    assert P.span_index(600.0, spans) == 0  # inside the first span
    assert P.span_index(1220.0, spans) == 0  # 20s past its end, within 30s -> nearest
    assert P.span_index(2385.0, spans) == 1  # 15s before the second span's start -> nearest
    assert P.span_index(5000.0, spans) is None  # far outside every span


def test_learn_offense_maps_by_span_with_no_spans_returns_no_maps():
    votes = [(0.0, P.RIGHT, 1), (1.0, P.LEFT, 0)]
    assert P.learn_offense_maps_by_span(votes, []) == []


def test_use_period_maps_is_false_for_none_and_empty_and_true_otherwise():
    assert P.use_period_maps(None) is False
    assert P.use_period_maps([]) is False
    assert P.use_period_maps([(0.0, 1200.0)]) is True


def make_period_frames(right_holder: int, start_t: float, n_per_half: int = 60, fps: int = 10):
    """One period's frames: `n_per_half` in the LEFT half, then `n_per_half` in the RIGHT half.

    Ten players, tracks 1-5 in cluster 0 and 6-10 in cluster 1, on a real (fittable) court
    homography. `right_holder` is the cluster carrying the ball while the action is in the RIGHT
    half; the other cluster carries it in the LEFT half - i.e. one period's fixed basket
    assignment. Times start at `start_t`, so two calls can be placed in different periods.
    """
    H = camera()
    verts = court.NCAA.vertices_array()
    kp = np.concatenate([to_pix(H, verts), np.ones((33, 1))], axis=1).tolist()
    rng = np.random.default_rng(7)
    frames = []
    for k in range(2 * n_per_half):
        half = -1 if k < n_per_half else 1
        holder = (1 - right_holder) if half == -1 else right_holder
        base_x = 20.0 if half == -1 else 74.0
        dets = []
        for tid in range(1, 11):
            cluster = 0 if tid <= 5 else 1
            x = base_x + rng.uniform(-8, 8)
            y = 5 + 4.5 * tid + rng.uniform(-1, 1)
            px, py = to_pix(H, [[x, y]])[0]
            cls = CLS_PLAYER_IN_POSSESSION if (cluster == holder and tid % 5 == 1) else CLS_PLAYER
            dets.append(Detection(track_id=tid, cls=cls, conf=0.9,
                                  bbox=[px - 15, py - 70, px + 15, py], team_cluster=cluster))
        t = start_t + k / fps
        frames.append(FrameRecord(frame_idx=round(t * 60), t=round(t, 3), keypoints=kp,
                                  detections=dets, ball=None, numbers=[]))
    return frames


def two_period_frames():
    """Two periods with inverted basket assignments (teams swap baskets at half time)."""
    return make_period_frames(right_holder=1, start_t=0.0) + \
        make_period_frames(right_holder=0, start_t=2400.0)


def test_build_possessions_inverts_offense_across_a_period_boundary_with_period_spans():
    # Period 1 [0, 1200): cluster 1 attacks RIGHT. Period 2 [2400, 3600): cluster 0 does. With
    # the period spans given, each possession's offense comes from its own period's map, so the
    # team attacking a given basket must invert across the boundary.
    frames = two_period_frames()
    spans = [(0.0, 1200.0), (2400.0, 3600.0)]

    possessions = extract.build_possessions(frames, fps=10, cluster_brightness={},
                                            period_spans=spans)

    by_period = {}
    for p in possessions:
        by_period.setdefault(0 if p.start_time < 1200 else 1, []).append(p)
    assert len(by_period[0]) == 2 and len(by_period[1]) == 2
    right_1 = next(p for p in by_period[0] if p.attacking_basket == "right")
    right_2 = next(p for p in by_period[1] if p.attacking_basket == "right")
    left_1 = next(p for p in by_period[0] if p.attacking_basket == "left")
    left_2 = next(p for p in by_period[1] if p.attacking_basket == "left")
    assert {right_1.offense_team, left_1.offense_team} == {DUKE, MICHIGAN}
    assert right_1.offense_team == left_2.offense_team
    assert right_2.offense_team == left_1.offense_team
    assert right_1.offense_team != right_2.offense_team  # the inversion at half time


def test_empty_period_spans_behaves_exactly_like_no_period_spans():
    # `use_period_maps([])` is False, so an OCR timeline that yielded no spans must fall back to
    # the per-segment rule rather than blanking (or guessing) every offense label.
    frames = two_period_frames()

    empty = extract.build_possessions(frames, fps=10, cluster_brightness={}, period_spans=[])
    none = extract.build_possessions(frames, fps=10, cluster_brightness={}, period_spans=None)

    assert [p.offense_team for p in empty] == [p.offense_team for p in none]
    assert all(p.offense_team is not None for p in empty)
