from basketball_plays import identity


def test_concurrent_duplicate_jerseys_keep_the_stronger_vote():
    tracks = [
        {"id": 1, "team": "Duke", "jersey": "12", "votes": 8, "start": 0.0, "end": 10.0},
        {"id": 2, "team": "Duke", "jersey": "12", "votes": 3, "start": 2.0, "end": 12.0},
        {"id": 3, "team": "Michigan", "jersey": "12", "votes": 3, "start": 2.0, "end": 12.0},
    ]
    keep = identity.resolve_conflicts(tracks)
    assert keep == {1: "12", 2: None, 3: "12"}


def test_sequential_tracks_may_share_a_jersey():
    tracks = [
        {"id": 1, "team": "Duke", "jersey": "12", "votes": 8, "start": 0.0, "end": 10.0},
        {"id": 2, "team": "Duke", "jersey": "12", "votes": 3, "start": 10.5, "end": 15.0},
    ]
    assert identity.resolve_conflicts(tracks) == {1: "12", 2: "12"}
