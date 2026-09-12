from basketball_plays import extract
from basketball_plays.schema import PlayerTrack


def track(tid, team, name, t0, n, x):
    rows = [[round(t0 + k / 10, 3), x, 25.0] for k in range(n)]
    return PlayerTrack(tid, team, None, name, rows, [])


def test_surplus_anonymous_short_track_is_demoted():
    players = [track(i, "Duke", f"P{i}", 0.0, 50, 10.0 + i) for i in range(5)]
    ghost = track(9, "Duke", None, 0.0, 30, 40.0)          # sixth Duke player for 3 s
    extract.enforce_team_cap(players + [ghost])
    assert ghost.team is None
    assert all(p.team == "Duke" for p in players)


def test_named_tracks_are_never_demoted_even_when_six_are_named():
    players = [track(i, "Duke", f"P{i}", 0.0, 50, 10.0 + i) for i in range(6)]
    extract.enforce_team_cap(players)
    assert all(p.team == "Duke" for p in players)


def test_track_only_briefly_surplus_keeps_its_team():
    players = [track(i, "Duke", f"P{i}", 0.0, 50, 10.0 + i) for i in range(5)]
    sub = track(7, "Duke", None, 4.0, 50, 40.0)             # overlaps the five for 1 s of 5 s
    extract.enforce_team_cap(players + [sub])
    assert sub.team == "Duke"


def test_opponent_tracks_are_counted_separately():
    duke = [track(i, "Duke", None, 0.0, 50, 10.0 + i) for i in range(5)]
    mich = [track(10 + i, "Michigan", None, 0.0, 50, 60.0 + i) for i in range(5)]
    extract.enforce_team_cap(duke + mich)
    assert all(p.team == "Duke" for p in duke) and all(p.team == "Michigan" for p in mich)
