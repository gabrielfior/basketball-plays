from basketball_plays import extract
from basketball_plays.schema import (
    CLS_PLAYER,
    CLS_PLAYER_IN_POSSESSION,
    Detection,
    FrameRecord,
    PlayerTrack,
    Possession,
)


def _frame(t, dets):
    return FrameRecord(frame_idx=int(t * 60), t=t, keypoints=[], detections=dets)


def test_attach_holding_marks_frames_where_the_track_had_the_ball():
    frames = [
        _frame(1.0, [Detection(7, CLS_PLAYER_IN_POSSESSION, 0.9, [100.0, 100.0, 140.0, 200.0]),
                     Detection(8, CLS_PLAYER, 0.9, [300.0, 100.0, 340.0, 200.0])]),
        _frame(1.1, [Detection(7, CLS_PLAYER, 0.9, [102.0, 100.0, 142.0, 200.0]),
                     Detection(8, CLS_PLAYER_IN_POSSESSION, 0.9, [301.0, 100.0, 341.0, 200.0])]),
    ]
    a = PlayerTrack(track_id=7, team="Duke", jersey=None, name=None,
                    trajectory=[[1.0, 10.0, 10.0], [1.1, 10.2, 10.0]],
                    boxes=[[1.0, 100.0, 100.0, 140.0, 200.0], [1.1, 102.0, 100.0, 142.0, 200.0]])
    b = PlayerTrack(track_id=8, team="Duke", jersey=None, name=None,
                    trajectory=[[1.0, 30.0, 10.0], [1.1, 30.1, 10.0]],
                    boxes=[[1.0, 300.0, 100.0, 340.0, 200.0], [1.1, 301.0, 100.0, 341.0, 200.0]])
    extract.attach_holding([a, b], frames)
    assert a.holding == [1.0]
    assert b.holding == [1.1]


def test_attach_holding_matches_rounded_boxes():
    # boxes in trajectories.jsonl are rounded to one decimal; detections are not
    frames = [_frame(2.0, [Detection(7, CLS_PLAYER_IN_POSSESSION, 0.9, [100.04, 99.96, 140.0, 200.0])])]
    a = PlayerTrack(7, "Duke", None, None, [[2.0, 1.0, 1.0]], [[2.0, 100.0, 100.0, 140.0, 200.0]])
    extract.attach_holding([a], frames)
    assert a.holding == [2.0]


def test_possession_json_roundtrip_keeps_holding_and_defaults_when_missing():
    p = PlayerTrack(7, "Duke", "12", "Cameron Boozer", [[1.0, 1.0, 1.0]], [[1.0, 0, 0, 1, 1]], holding=[1.0])
    pos = Possession(0, 1.0, 2.0, 10, "Duke", "left", [p], [])
    back = Possession.from_dict(__import__("json").loads(pos.to_json()))
    assert back.players[0].holding == [1.0]
    d = __import__("json").loads(pos.to_json())
    del d["players"][0]["holding"]
    assert Possession.from_dict(d).players[0].holding == []
