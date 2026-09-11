from basketball_plays import schema


def test_frame_record_roundtrip(tmp_path):
    fr = schema.FrameRecord(
        frame_idx=3, t=0.3, keypoints=[[1.0, 2.0, 0.9]] * 33,
        detections=[schema.Detection(track_id=1, cls=3, conf=0.8, bbox=[1, 2, 3, 4],
                                     team_cluster=0)],
        ball=[5, 6, 7, 8], numbers=[{"track_id": 1, "text": "12"}], shot=True,
    )
    p = tmp_path / "frames.jsonl"
    schema.write_jsonl(p, [fr])
    back = schema.read_frames(p)
    assert back == [fr]


def test_possession_roundtrip(tmp_path):
    pos = schema.Possession(
        possession_id=0, start_time=1.0, end_time=4.0, fps=10, offense_team="Duke",
        attacking_basket="left",
        players=[schema.PlayerTrack(track_id=1, team="Duke", jersey="12", name="Cameron Boozer",
                                    trajectory=[[1.0, 10.0, 20.0]], boxes=[[1.0, 1, 2, 3, 4]])],
        ball=[[1.0, 11.0, 21.0]],
    )
    p = tmp_path / "trajectories.jsonl"
    schema.write_jsonl(p, [pos])
    assert schema.read_possessions(p) == [pos]
