from basketball_plays import halfcourt as H
from basketball_plays.playbyplay import Event
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


def test_clock_to_video_outside_the_timeline_is_none():
    rs = reads([(10, 1000), (11, 999)])
    assert H.clock_to_video(rs, 1100, "first") is None
    assert H.clock_to_video(rs, 900, "last") is None
