import cv2
import numpy as np

from basketball_plays import court, extract
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import (
    CLS_NUMBER,
    CLS_PLAYER,
    CLS_PLAYER_IN_POSSESSION,
    CLS_REFEREE,
    Detection,
    FrameRecord,
)

FPS = 10


def camera():
    src = np.array([[0, 0], [94, 0], [94, 50], [0, 50]], dtype=np.float32)
    dst = np.array([[200, 150], [1080, 150], [1250, 700], [30, 700]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def to_pix(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def make_frames(n_left=60, n_right=60, n_invalid=0):
    """10 players: tracks 1-5 cluster 0, 6-10 cluster 1. Left half first, then right half.

    Cluster 1 has the ball on the left, cluster 0 on the right. Track 1 wears Duke #12,
    track 6 Michigan #23.
    """
    H = camera()
    verts = court.NCAA.vertices_array()
    kp = np.concatenate([to_pix(H, verts), np.ones((33, 1))], axis=1).tolist()
    frames = []
    rng = np.random.default_rng(1)
    fi = 0

    def frame(t_idx, half, valid=True):
        nonlocal fi
        dets, numbers = [], []
        base_x = 20.0 if half == -1 else 74.0
        for tid in range(1, 11):
            cluster = 0 if tid <= 5 else 1
            x = base_x + rng.uniform(-8, 8) + 0.1 * t_idx
            y = 5 + 4.5 * tid + rng.uniform(-1, 1)
            px, py = to_pix(H, [[x, y]])[0]
            cls = CLS_PLAYER
            if (half == -1 and tid == 6) or (half == 1 and tid == 1):
                cls = CLS_PLAYER_IN_POSSESSION
            dets.append(Detection(track_id=tid, cls=cls, conf=0.9,
                                  bbox=[px - 15, py - 70, px + 15, py], team_cluster=cluster))
        dets.append(Detection(track_id=99, cls=CLS_REFEREE, conf=0.9, bbox=[600, 300, 630, 380]))
        dets.append(Detection(track_id=98, cls=CLS_NUMBER, conf=0.9, bbox=[600, 300, 630, 380]))
        if t_idx % 10 == 0:
            numbers = [{"track_id": 1, "text": "12"}, {"track_id": 6, "text": "23"},
                       {"track_id": 2, "text": "7"}]  # track 2 read only once per 10 -> few votes
        ball_px = to_pix(H, [[base_x, 25.0]])[0]
        rec = FrameRecord(frame_idx=fi * 6, t=fi / FPS,
                          keypoints=kp if valid else [[0, 0, 0.0]] * 33,
                          detections=dets if valid else [],
                          ball=[ball_px[0] - 5, ball_px[1] - 5, ball_px[0] + 5, ball_px[1] + 5] if valid else None,
                          numbers=numbers if valid else [])
        fi += 1
        return rec

    for k in range(n_left):
        frames.append(frame(k, -1))
    for k in range(n_invalid):
        frames.append(frame(k, -1, valid=False))
    for k in range(n_right):
        frames.append(frame(k, 1))
    return frames


def test_two_possessions_with_teams_and_names():
    frames = make_frames()
    poss = extract.build_possessions(frames, fps=FPS, cluster_brightness={0: 220.0, 1: 90.0})
    assert len(poss) == 2
    a, b = poss
    assert a.attacking_basket == "left" and b.attacking_basket == "right"
    assert a.start_time == 0.0 and abs(a.end_time - 6.0) < 1e-6
    assert abs(b.start_time - 6.0) < 1e-6
    # cluster 0 is bright -> Duke; cluster 1 had the ball on the left
    assert a.offense_team == MICHIGAN and b.offense_team == DUKE
    assert len(a.players) == 10 and len(b.players) == 10
    by_id = {p.track_id: p for p in a.players}
    assert by_id[1].team == DUKE and by_id[1].jersey == "12" and by_id[1].name == "Cameron Boozer"
    assert by_id[6].team == MICHIGAN and by_id[6].jersey == "23" and by_id[6].name == "Yaxel Lendeborg"
    assert by_id[3].jersey is None and by_id[3].team == DUKE
    # trajectories are in court feet on the correct half and boxes keep pixel coordinates
    xs = np.array([p[1] for p in by_id[1].trajectory])
    assert (xs < 47).all() and len(by_id[1].trajectory) == 60
    assert len(by_id[1].boxes) == 60 and by_id[1].boxes[0][3] > 100
    assert len(a.ball) == 60 and all(x < 47 for _, x, _ in a.ball)


def test_invalid_stretch_is_excluded_and_splits():
    frames = make_frames(n_left=60, n_invalid=40, n_right=60)
    poss = extract.build_possessions(frames, fps=FPS, cluster_brightness={0: 220.0, 1: 90.0})
    assert len(poss) == 2
    assert abs(poss[0].end_time - 6.0) < 1e-6
    assert abs(poss[1].start_time - 10.0) < 1e-6


def test_team_map_override():
    frames = make_frames()
    poss = extract.build_possessions(frames, fps=FPS, team_map={0: MICHIGAN, 1: DUKE})
    assert poss[0].offense_team == DUKE


def test_referees_and_numbers_are_not_players():
    frames = make_frames(n_left=40, n_right=0)
    poss = extract.build_possessions(frames, fps=FPS, cluster_brightness={0: 220.0, 1: 90.0})
    assert len(poss) == 1
    assert {p.track_id for p in poss[0].players} == set(range(1, 11))
