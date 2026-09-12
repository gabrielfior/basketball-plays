import json

import pytest

from basketball_plays import halfcourt as H
from basketball_plays.playbyplay import Event
from basketball_plays.schema import PlayerTrack, Possession
from basketball_plays.scoreboard import ScoreboardRead

D, M = "Duke", "Michigan"


def ev(clock_text, team, etype, text):
    m, s = clock_text.split(":")
    return Event(clock_text=clock_text, clock=int(m) * 60 + int(s), team=team, type=etype, text=text,
                 scoring=" makes " in text, score_value=0, away_score=0, home_score=0)


# Michigan at Duke 2026-02-21, first 4:14 of play-by-play (ESPN 401817238)
OPENING = [
    ev("20:00", None, "Jumpball", "Start game"),
    ev("19:59", M, "Jumpball", "Jump Ball won by Michigan"),
    ev("19:59", D, "Jumpball", "Jump Ball lost by Duke"),
    ev("19:32", M, "DunkShot", "Aday Mara makes 3-foot alley oop dunk"),
    ev("19:14", D, "JumpShot", "Isaiah Evans misses 24-foot three point jumper"),
    ev("19:11", D, "Offensive Rebound", "Patrick Ngongba II Offensive Rebound."),
    ev("19:10", D, "JumpShot", "Isaiah Evans misses 24-foot three point jumper"),
    ev("19:07", M, "Defensive Rebound", "Aday Mara Defensive Rebound."),
    ev("19:02", D, "PersonalFoul", "Foul on Isaiah Evans."),
    ev("18:59", M, "JumpShot", "Morez Johnson Jr. makes 11-foot pullup jump shot (Aday Mara assists)"),
    ev("18:40", D, "DunkShot", "Isaiah Evans makes 1-foot dunk (Caleb Foster assists)"),
    ev("18:19", M, "JumpShot", "Elliot Cadeau misses 26-foot three point jumper"),
    ev("18:14", D, "Defensive Rebound", "Isaiah Evans Defensive Rebound."),
    ev("17:59", D, "JumpShot", "Isaiah Evans misses 23-foot three point jumper"),
    ev("17:56", D, "Offensive Rebound", "Cameron Boozer Offensive Rebound."),
    ev("17:55", M, "PersonalFoul", "Foul on Aday Mara."),
    ev("17:42", D, "JumpShot", "Dame Sarr makes 23-foot three point jumper (Cameron Boozer assists)"),
    ev("17:28", M, "JumpShot", "Yaxel Lendeborg makes 12-foot pullup jump shot"),
    ev("17:28", D, "PersonalFoul", "Foul on Caleb Foster."),
    ev("17:28", M, "Substitution", "Aday Mara subbing out for Michigan"),
    ev("17:28", M, "Substitution", "Roddy Gayle Jr. subbing in for Michigan"),
    ev("17:28", M, "MadeFreeThrow", "Yaxel Lendeborg misses free throw 1 of 1"),
    ev("17:27", D, "Defensive Rebound", "Cameron Boozer Defensive Rebound."),
    ev("17:14", M, "PersonalFoul", "Foul on Nimari Burnett."),
    ev("17:09", D, "Lost Ball Turnover", "Caleb Foster bad pass\nturnover"),
    ev("17:09", M, "Steal", "Roddy Gayle Jr. Steal."),
    ev("17:03", M, "LayUpShot", "Yaxel Lendeborg makes layup"),
    ev("16:53", M, "PersonalFoul", "Foul on Morez Johnson Jr.."),
    ev("16:53", D, "MadeFreeThrow", "Caleb Foster makes free throw 1 of 2"),
    ev("16:53", D, "MadeFreeThrow", "Caleb Foster makes free throw 2 of 2"),
    ev("16:41", M, "JumpShot", "Elliot Cadeau misses 26-foot three point jumper"),
    ev("16:38", D, "Defensive Rebound", "Caleb Foster Defensive Rebound."),
    ev("16:30", D, "JumpShot", "Isaiah Evans makes 23-foot three point jumper"),
    ev("16:24", M, "JumpShot", "Yaxel Lendeborg makes 10-foot floating jump shot (Elliot Cadeau assists)"),
    ev("16:24", D, "PersonalFoul", "Foul on Cameron Boozer."),
    ev("16:24", D, "Substitution", "Patrick Ngongba II subbing out for Duke"),
    ev("16:24", D, "Substitution", "Maliq Brown subbing in for Duke"),
    ev("16:24", M, "MadeFreeThrow", "Yaxel Lendeborg makes free throw 1 of 1"),
    ev("15:56", D, "DunkShot", "Maliq Brown makes 1-foot dunk (Cameron Boozer assists)"),
    ev("15:46", None, "OfficialTVTimeOut", "Official TV Timeout"),
    ev("15:46", D, "Substitution", "Caleb Foster subbing out for Duke"),
    ev("15:46", D, "Substitution", "Cayden Boozer subbing in for Duke"),
    ev("15:30", M, "JumpShot", "L.J. Cason misses 24-foot three point jumper"),
    ev("15:27", D, "Defensive Rebound", "Isaiah Evans Defensive Rebound."),
]


def summary(iv):
    return (iv.team, iv.start_clock, iv.end_clock, iv.start_type, iv.terminal, iv.free_throws)


def test_duke_intervals_in_the_opening_sequence():
    ivs = [summary(i) for i in H.intervals(OPENING) if i.team == D]
    assert ivs == [
        (D, 1172, 1150, "dead", "shot", False),      # after Mara's dunk, two misses, Michigan rebounds
        (D, 1139, 1120, "dead", "shot", False),      # Evans dunk
        (D, 1094, 1075, "live", "stoppage", False),  # def reb, miss, off reb, foul on Mara stops play
        (D, 1075, 1062, "dead", "shot", False),      # sideline inbound, Sarr three
        (D, 1047, 1034, "live", "stoppage", False),  # def reb after missed FT, foul on Burnett
        (D, 1034, 1029, "dead", "turnover", False),  # Foster bad pass
        (D, 1023, 1013, "dead", "foul", True),       # fouled, two free throws
        (D, 998, 990, "live", "shot", False),        # def reb, Evans three
        (D, 984, 956, "dead", "shot", False),        # after Lendeborg and-one, Brown dunk
    ]


def test_duke_full_court_starts_follow_a_michigan_score():
    duke_intervals = [i for i in H.intervals(OPENING) if i.team == D]
    # True exactly where Duke inbounds under its own basket after Michigan scored: Mara's dunk
    # at 19:32, Johnson's jumper at 18:59, Lendeborg's layup at 17:03 and Lendeborg's made
    # and-one free throw at 16:24. The 1075 start is a sideline inbound after the foul on Mara
    # stopped a Duke frontcourt possession; the other starts are fouls or defensive rebounds.
    assert [i.full_court for i in duke_intervals] == [
        True, True, False, False, False, False, True, False, True,
    ]
    assert [i.start_clock for i in duke_intervals if i.full_court] == [1172, 1139, 1023, 984]


def test_michigan_gets_an_ato_interval_after_the_tv_timeout():
    ivs = [summary(i) for i in H.intervals(OPENING) if i.team == M]
    assert ivs[-1] == (M, 946, 930, "ato", "shot", False)
    # the stretch before the timeout is its own interval ending in a stoppage
    assert ivs[-2] == (M, 956, 946, "dead", "stoppage", False)


def test_first_interval_starts_live_from_the_jump_ball():
    first = H.intervals(OPENING)[0]
    assert (first.team, first.start_clock, first.start_type) == (M, 1199, "live")


def test_and_one_does_not_create_an_empty_interval_for_the_fouling_team():
    # 17:28: Michigan scores, Duke fouls, Michigan shoots one free throw. No Duke interval at 1048.
    assert not any(i.team == D and i.start_clock == 1048 for i in H.intervals(OPENING))


def test_zero_duration_intervals_are_dropped():
    for iv in H.intervals(OPENING):
        assert iv.end_clock < iv.start_clock


def test_period_end_closes_an_open_interval():
    events = [
        ev("20:00", None, "Jumpball", "Start game"),
        ev("19:59", D, "Jumpball", "Jump Ball won by Duke"),
        ev("0:00", None, "End Period", "End of 1st Half"),
    ]
    ivs = [summary(i) for i in H.intervals(events)]
    assert ivs == [(D, 1199, 0.0, "live", "period_end", False)]


def test_offensive_foul_turns_the_ball_over():
    events = [
        ev("20:00", None, "Jumpball", "Start game"),
        ev("19:59", D, "Jumpball", "Jump Ball won by Duke"),
        ev("10:00", D, "PersonalFoul", "Foul on Cameron Boozer."),
        ev("9:50", M, "JumpShot", "Yaxel Lendeborg makes 10-foot jumper"),
    ]
    ivs = [summary(i) for i in H.intervals(events)]
    assert ivs == [
        (D, 1199, 600, "live", "turnover", False),
        (M, 600, 590, "dead", "shot", False),
    ]


def test_turnover_without_a_steal_opens_a_dead_ball_interval():
    events = [
        ev("20:00", None, "Jumpball", "Start game"),
        ev("19:59", D, "Jumpball", "Jump Ball won by Duke"),
        ev("9:00", D, "Lost Ball Turnover", "Cameron Boozer traveling turnover"),
        ev("8:50", M, "JumpShot", "Yaxel Lendeborg makes 10-foot jumper"),
    ]
    ivs = [summary(i) for i in H.intervals(events)]
    assert ivs == [
        (D, 1199, 540, "live", "turnover", False),
        (M, 540, 530, "dead", "shot", False),
    ]


def test_dead_ball_rebound_closes_the_shot_and_opens_dead_ball():
    events = [
        ev("20:00", None, "Jumpball", "Start game"),
        ev("19:59", D, "Jumpball", "Jump Ball won by Duke"),
        ev("8:40", D, "JumpShot", "Isaiah Evans misses 24-foot three point jumper"),
        ev("8:30", M, "Dead Ball Rebound", "Michigan Dead Ball Rebound."),
        ev("8:10", M, "JumpShot", "Yaxel Lendeborg makes 10-foot jumper"),
    ]
    ivs = [summary(i) for i in H.intervals(events)]
    assert ivs == [
        (D, 1199, 520, "live", "shot", False),
        (M, 510, 490, "dead", "shot", False),
    ]


def test_offensive_rebound_with_no_open_interval_starts_live():
    events = [
        ev("7:10", M, "MadeFreeThrow", "Yaxel Lendeborg misses free throw 1 of 1"),
        ev("7:00", M, "Offensive Rebound", "Aday Mara Offensive Rebound."),
        ev("6:50", M, "JumpShot", "Aday Mara makes 5-foot jumper"),
    ]
    ivs = [summary(i) for i in H.intervals(events)]
    assert ivs == [(M, 420, 410, "live", "shot", False)]


def test_technical_foul_keeps_possession_through_the_free_throw():
    events = [
        ev("20:00", None, "Jumpball", "Start game"),
        ev("19:59", D, "Jumpball", "Jump Ball won by Duke"),
        ev("6:00", M, "TechnicalFoul", "Technical foul on Michigan bench"),
        ev("6:00", D, "MadeFreeThrow", "Cameron Boozer makes technical free throw 1 of 1"),
        ev("5:50", D, "JumpShot", "Cameron Boozer makes 10-foot jumper"),
    ]
    ivs = [summary(i) for i in H.intervals(events) if i.team == D]
    assert ivs[0] == (D, 1199, 360, "live", "stoppage", False)
    assert ivs[1] == (D, 360, 350, "dead", "shot", False)


def test_first_shot_with_no_jump_ball_opens_a_period_interval():
    events = [ev("19:40", D, "JumpShot", "Cameron Boozer makes 10-foot jumper")]
    first = H.intervals(events)[0]
    assert summary(first) == (D, 1200, 1180, "period", "shot", False)


def reads(pairs):
    return [ScoreboardRead(t=float(t), clock=c, clock_text=None, away=None, home=None)
            for t, c in pairs]


def test_clock_to_video_first_and_last_read_of_a_stopped_clock():
    rs = reads([(10, 1000), (11, 999), (12, 998), (13, 998), (14, 998), (15, 998),
                (16, 997)])
    assert H.clock_to_video(rs, 998, "first") == 12.0
    assert H.clock_to_video(rs, 998, "last") == 15.0
    assert H.clock_to_video(rs, 997, "first") == 16.0


def test_clock_to_video_interpolates_across_missing_reads():
    rs = reads([(10, 1000), (11, None), (12, None), (13, 997)])
    # 999 lies one third of the way from 1000 to 997
    assert H.clock_to_video(rs, 999, "first") == 11.0
    assert H.clock_to_video(rs, 998, "last") == 12.0


def test_clock_to_video_refuses_to_bridge_a_replay():
    rs = reads([(10, 1000), (60, 990)])
    assert H.clock_to_video(rs, 995, "first") is None
    assert H.clock_to_video(rs, 995, "first", max_gap_s=100) == 35.0


def test_clock_to_video_restricts_reads_to_the_period_window():
    # the same game clock comes round again in the second half
    rs = reads([(99, 601), (100, 600), (101, 599), (2499, 601), (2500, 600), (2501, 599)])
    assert H.clock_to_video(rs, 600, "last") == 2500.0
    assert H.clock_to_video(rs, 600, "last", t_hi=1500) == 100.0
    assert H.clock_to_video(rs, 600, "first", t_lo=1500) == 2500.0


def test_clock_to_video_outside_the_timeline_is_none():
    rs = reads([(10, 1000), (11, 999)])
    assert H.clock_to_video(rs, 1100, "first") is None
    assert H.clock_to_video(rs, 900, "last") is None


def traj(t_start, n, x, y, dx=0.0, dy=0.0, fps=10.0):
    """[t, x, y] rows moving by (dx, dy) ft per frame."""
    return [[round(t_start + k / fps, 3), x + k * dx, y + k * dy] for k in range(n)]


def make_possession(players, basket="left", pid=0, t0=100.0, t1=120.0):
    return Possession(pid, t0, t1, 10.0, "Duke", basket, players, [])


def test_gather_tracks_mirrors_right_basket_and_filters_team_and_window():
    duke = PlayerTrack(1, "Duke", "12", "Cameron Boozer", traj(100.0, 50, 70.0, 10.0), [],
                       holding=[101.0])
    mich = PlayerTrack(2, "Michigan", None, None, traj(100.0, 50, 60.0, 10.0), [])
    pos = make_possession([duke, mich], basket="right")
    tracks = H.gather_tracks([pos], "Duke", 100.5, 102.0, "right")
    assert [t.track_id for t in tracks] == [1]
    assert min(tracks[0].xy) == 100.5 and max(tracks[0].xy) == 102.0
    assert tracks[0].xy[100.5] == (24.0, 10.0)          # 94 - 70
    assert tracks[0].holding == {101.0}
    assert tracks[0].name == "Cameron Boozer"


def test_gather_tracks_spans_two_vision_possessions_and_skips_nan():
    a = PlayerTrack(1, "Duke", None, None, traj(100.0, 20, 30.0, 10.0), [])
    b = PlayerTrack(5, "Duke", None, None,
                    [[102.0, float("nan"), float("nan")], [102.1, 31.0, 10.0]], [])
    tracks = H.gather_tracks(
        [make_possession([a], t0=100, t1=102), make_possession([b], pid=1, t0=102, t1=103)],
        "Duke", 100.0, 103.0, "left")
    assert sorted(t.track_id for t in tracks) == [1, 5]
    assert 102.0 not in next(t for t in tracks if t.track_id == 5).xy


def test_attack_direction_is_the_majority_over_the_team_own_possessions():
    duke = [make_possession([], basket="right", pid=i) for i in range(3)]
    duke[2].attacking_basket = "left"                      # one mislabelled vision possession
    mich = make_possession([], basket="left", pid=9)
    mich.offense_team = "Michigan"
    assert H.attack_direction(duke + [mich], "Duke") == "right"
    assert H.attack_direction(duke + [mich], "Michigan") == "left"
    with pytest.raises(ValueError):
        H.attack_direction(duke, "Villanova")


def test_gather_tracks_mirrors_by_the_team_direction_not_the_possession():
    # a Duke player standing at raw x = 70 across two vision possessions: Duke's own (attacking
    # right) and Michigan's (attacking left, where the Duke track is a defender). Both must
    # mirror the same way, by Duke's direction, or the second one lands 180 degrees out.
    rows = traj(100.0, 20, 70.0, 10.0)
    duke_pos = make_possession(
        [PlayerTrack(1, "Duke", "12", None, rows[:10], [])], basket="right", t0=100.0, t1=100.9)
    mich_pos = make_possession(
        [PlayerTrack(1, "Duke", "12", None, rows[10:], [])], basket="left", pid=1,
        t0=101.0, t1=101.9)
    mich_pos.offense_team = "Michigan"
    possessions = [duke_pos, mich_pos]
    tracks = H.gather_tracks(possessions, "Duke", 100.0, 101.9,
                             H.attack_direction(possessions, "Duke"))
    assert [x for tr in tracks for x, _ in tr.xy.values()] == [24.0] * 20


def test_ball_handler_series_uses_holding_times():
    a = H.Track(1, None, None, {100.0: (60.0, 25.0), 100.1: (58.0, 25.0)}, {100.1})
    b = H.Track(2, None, None, {100.0: (40.0, 25.0)}, {100.0})
    assert H.ball_handler_series([a, b]) == {100.0: (40.0, 25.0), 100.1: (58.0, 25.0)}


def test_find_t0_is_when_the_handler_crosses_half_court():
    handler = {100.0: (60.0, 25.0), 100.1: (50.0, 25.0), 100.2: (46.0, 25.0), 100.3: (40.0, 25.0)}
    assert H.find_t0([], handler, 100.0, 110.0) == 100.2


def test_find_t0_falls_back_to_the_team_centroid():
    tracks = [H.Track(i, None, None, {100.0: (60.0, 10.0 * i), 100.5: (30.0, 10.0 * i)}, set())
              for i in range(1, 4)]
    assert H.find_t0(tracks, {}, 100.0, 110.0) == 100.5
    assert H.find_t0(tracks[:2], {}, 100.0, 110.0) is None   # fewer than three players visible


def _still_tracks(t_start, n_frames, moving=False, n=5):
    out = []
    for i in range(n):
        dx = 0.4 if moving else 0.02  # 4 ft/s vs 0.2 ft/s
        rows = traj(t_start, n_frames, 20.0 + 3 * i, 5.0 + 8 * i, dx=dx)
        out.append(H.Track(i, None, None, {r[0]: (r[1], r[2]) for r in rows}, set()))
    return out


def test_still_players_counts_visible_and_still():
    assert H.still_players(_still_tracks(100.0, 10), 100.9) == (5, 5)
    assert H.still_players(_still_tracks(100.0, 10, moving=True), 100.9) == (5, 0)
    assert H.still_players(_still_tracks(100.0, 10), 100.2) == (5, 0)   # not enough history yet


def test_still_players_ignores_a_duplicate_track_id_for_the_same_player():
    tracks = _still_tracks(100.0, 10)
    dup = H.Track(99, None, None, dict(tracks[0].xy), set())  # same player, a second track id
    assert H.still_players(tracks + [dup], 100.9) == (5, 5)


def test_still_players_survives_a_track_break():
    others = _still_tracks(100.0, 20)[1:]  # players 1..4, still from 100.0 to 101.9
    rows = traj(100.0, 20, 20.0, 5.0, dx=0.0)  # player 0, stationary
    a = H.Track(0, None, None, {r[0]: (r[1], r[2]) for r in rows if r[0] <= 100.4}, set())
    b = H.Track(50, None, None, {r[0]: (r[1], r[2]) for r in rows if r[0] >= 100.5}, set())
    assert H.still_players([a, b] + others, 100.9) == (5, 5)


def test_still_players_tolerates_a_dropped_frame_at_the_lookback_time():
    tracks = _still_tracks(100.0, 10)
    gapped = [H.Track(tr.track_id, None, None,
                      {t: p for t, p in tr.xy.items() if abs(t - 100.4) > 1e-6}, set())
              for tr in tracks]
    assert all(100.4 not in tr.xy for tr in gapped)
    assert H.still_players(gapped, 100.9) == (5, 5)


def test_find_setup_dead_ball_takes_the_last_still_frame_before_the_inbound():
    tracks = _still_tracks(96.0, 60)  # still from 96.0 to 101.9
    setup, no_setup = H.find_setup(tracks, {}, "dead", t_start=100.0, t0=100.4, t_end=115.0)
    assert (setup, no_setup) == (101.5, False)   # end of the [t_start - 3, t_start + 1.5] search


def test_find_setup_live_takes_the_first_still_frame_with_handler_beyond_the_arc():
    moving = _still_tracks(100.0, 20, moving=True)          # 100.0 .. 101.9 moving
    still = _still_tracks(102.0, 40)                        # 102.0 .. 105.9 still
    tracks = [H.Track(i, None, None, {**moving[i].xy, **still[i].xy}, set()) for i in range(5)]
    handler = {t: (30.0, 25.0) for t in still[0].xy}        # 24.75 ft from the rim
    setup, no_setup = H.find_setup(tracks, handler, "live", t_start=99.0, t0=100.0, t_end=115.0)
    assert (setup, no_setup) == (102.5, False)
    inside = {t: (12.0, 25.0) for t in still[0].xy}
    assert H.find_setup(tracks, inside, "live", 99.0, 100.0, 115.0) == (100.0, True)


def test_find_setup_rejects_a_still_backcourt_lineup():
    # five players standing around x = 70: still, but waiting to inbound in their own backcourt
    tracks = [H.Track(i, None, None,
                      {r[0]: (r[1], r[2]) for r in traj(96.0, 60, 68.0 + i, 5.0 + 8 * i)}, set())
              for i in range(5)]
    assert H.find_setup(tracks, {}, "dead", t_start=100.0, t0=100.4, t_end=115.0) == (100.4, True)
    assert H.find_setup(tracks, {}, "live", 99.0, 100.4, 115.0) == (100.4, True)


def test_find_setup_full_court_dead_start_uses_the_live_rule_from_t0():
    # backcourt and moving until 101.9, then set up in the frontcourt from 102.0
    back = [H.Track(i, None, None,
                    {r[0]: (r[1], r[2]) for r in traj(97.0, 50, 60.0 + 3 * i, 5.0 + 8 * i, dx=0.4)},
                    set()) for i in range(5)]
    front = _still_tracks(102.0, 40)
    tracks = [H.Track(i, None, None, {**back[i].xy, **front[i].xy}, set()) for i in range(5)]
    # the dead-ball window [97, 101.5] only holds the backcourt lineup, so it must be skipped
    assert H.find_setup(tracks, {}, "dead", 100.0, 102.0, 115.0, full_court=True) == (102.5, False)


def test_find_setup_without_a_still_frame_flags_no_setup():
    tracks = _still_tracks(100.0, 60, moving=True)
    assert H.find_setup(tracks, {}, "live", 99.0, 100.0, 115.0) == (100.0, True)
    assert H.find_setup(tracks, {}, "ato", 99.0, 100.0, 115.0) == (100.0, True)


def test_build_records_end_to_end_on_a_synthetic_possession():
    # Michigan scores at 19:32 (clock 1172), Duke inbounds, walks it up, Evans misses at 19:14
    # (1154), Michigan rebounds at 19:07. Scoreboard: clock 1172 shown at video 60..63 s, then runs.
    events = OPENING[:8]
    rs = reads([(58, 1174), (59, 1173), (60, 1172), (61, 1172), (62, 1172), (63, 1172)]
               + [(63 + k, 1171 - (k - 1)) for k in range(1, 30)])
    # Duke tracks: five players walking up from 60 s, still from 66 s to 70 s, at the right basket
    players = []
    for i in range(5):
        # moving up court (mirrored later)
        rows = traj(60.0, 60, 30.0 + 2 * i, 8.0 + 8 * i, dx=0.3)
        rows += traj(66.0, 40, rows[-1][1], rows[-1][2])                    # still
        players.append(
            PlayerTrack(10 + i, "Duke", None, None, rows, [], holding=[66.5] if i == 0 else [])
        )
    # the vision possession list spans the period: it bounds the scoreboard reads a clock may
    # map to (the same clock comes round again in the second half)
    pos = make_possession(players, basket="right", t0=60.0, t1=95.0)
    recs = H.build_records("test", "Duke", events, rs, [pos])
    assert len(recs) == 1
    r = recs[0]
    assert (r.start_type, r.terminal, r.clock_start, r.clock_end) == ("dead", "shot", 1172, 1150)
    assert r.full_court is True and r.suspect_duplicates is False
    assert r.located and r.t_start == 63.0                    # last read showing 19:32 -> inbound
    # half a second of slack past the read
    assert r.t_end == H.clock_to_video(rs, 1150, "first") + 0.5
    assert r.t0 is not None and r.setup is not None and not r.no_setup and not r.transition
    assert r.outcome == "missed_3"
    assert r.n_visible_at_setup == 5
    assert len(r.players) == 5 and all(
        row[1] < 47 for p in r.players for row in p["trajectory"][-5:]
    )
    back = H.HalfcourtRecord.from_dict(json.loads(r.to_json()))
    assert back.setup == r.setup
    assert [e["type"] for e in back.events] == [
        "JumpShot", "Offensive Rebound", "JumpShot", "Defensive Rebound",
    ]


def test_build_records_marks_unlocatable_intervals():
    events = OPENING[:8]
    rs = reads([(10, 1199), (11, 1198)])                      # timeline ends long before 1172
    recs = H.build_records("test", "Duke", events, rs, [])
    assert (
        len(recs) == 1
        and not recs[0].located and recs[0].t_start is None and recs[0].players == []
    )


def test_build_records_located_without_a_track_reaching_the_frontcourt():
    events = OPENING[:8]
    rs = reads([(58, 1174), (59, 1173), (60, 1172), (61, 1172), (62, 1172), (63, 1172)]
               + [(63 + k, 1171 - (k - 1)) for k in range(1, 30)])
    # five Duke players stuck at raw x = 20.0 at the right basket -> mirrors to x = 74, backcourt
    players = [
        PlayerTrack(10 + i, "Duke", None, None, traj(60.0, 260, 20.0, 8.0 + 8 * i), [])
        for i in range(5)
    ]
    pos = make_possession(players, basket="right", t0=60.0, t1=90.0)
    recs = H.build_records("test", "Duke", events, rs, [pos])
    assert len(recs) == 1
    r = recs[0]
    assert r.located is True
    assert r.t_start == 63.0
    assert r.t_end is not None
    assert r.t0 is None and r.setup is None and r.no_setup is True and r.transition is False
    assert r.players == [] and r.ball_handler == []
    assert r.outcome == "missed_3"
    assert r.n_visible_at_setup == 0


def test_default_stillness_threshold_is_one_and_a_half_feet():
    # 1.2 ft over 0.5 s is above the 1 ft merge radius but under the 1.5 ft stillness threshold
    assert H.STILL_FT == 1.5 and H.MERGE_FT == 1.0
    tracks = []
    for i in range(5):
        rows = traj(100.0, 10, 20.0 + 3 * i, 5.0 + 8 * i, dx=0.24)  # 0.24 ft/frame = 1.2 ft/0.5 s
        tracks.append(H.Track(i, None, None, {r[0]: (r[1], r[2]) for r in rows}, set()))
    assert H.still_players(tracks, 100.9) == (5, 5)
    assert H.still_players(tracks, 100.9, max_move_ft=1.0) == (5, 0)


def test_positions_at_caps_at_five_keeping_detected_identified_long_tracks():
    key = 100.0
    mk = lambda i, name, det, n, x: H.Track(i, name, None, {key: (x, 25.0)}, set(),
                                           detected={key} if det else set(), length=n)
    tracks = [mk(1, "A", True, 50, 10.0), mk(2, "B", True, 50, 15.0), mk(3, "C", True, 50, 20.0),
              mk(4, None, True, 40, 25.0), mk(5, "D", False, 50, 30.0),   # D interpolated
              mk(6, None, True, 5, 35.0), mk(7, None, False, 3, 40.0)]    # anonymous extras
    kept = H.positions_at(tracks, key)
    assert len(kept) == 5
    xs = sorted(x for x, _ in kept)
    # detection beats identity: the anonymous detected tracks 4 and 6 stay, the interpolated
    # named D and the interpolated anonymous 7 are dropped
    assert xs == [10.0, 15.0, 20.0, 25.0, 35.0]
    assert H.merged_count(tracks, key) == 7
    assert len(H.positions_at(tracks, key, cap=None)) == 7


def test_positions_at_keeps_the_better_ranked_member_of_a_duplicate_pair():
    key = 100.0
    good = H.Track(1, "A", None, {key: (10.0, 25.0)}, set(), detected={key}, length=50)
    ghost = H.Track(2, None, None, {key: (10.5, 25.0)}, set(), detected=set(), length=4)
    assert H.positions_at([ghost, good], key) == [(10.0, 25.0)]


def test_tracks_from_record_roundtrips_detected_and_length():
    rec = H.HalfcourtRecord(
        game_id="g", index=0, team="Duke", start_type="dead", terminal="shot", free_throws=False,
        clock_start=1000, clock_end=990, t_start=1.0, t_end=5.0, t0=1.0, setup=1.0,
        no_setup=False, transition=False, located=True, outcome=None, points=0,
        n_visible_at_setup=1,
        players=[{"track_id": 9, "name": "A", "jersey": "12", "trajectory": [[1.0, 2.0, 3.0]],
                  "detected": [1.0], "length": 7}],
    )
    tr = H.tracks_from_record(rec)[0]
    assert (tr.track_id, tr.name, tr.jersey, tr.detected, tr.length) == (9, "A", "12", {1.0}, 7)
    assert tr.xy == {1.0: (2.0, 3.0)}
