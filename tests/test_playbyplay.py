from basketball_plays import playbyplay as pbp


def ev(clock, text, team="150", type_="JumpShot", score=None, scoring=False):
    return {
        "clock": {"displayValue": clock}, "text": text, "team": {"id": team},
        "type": {"text": type_}, "scoringPlay": scoring, "scoreValue": score or 0,
        "period": {"number": 1}, "awayScore": 0, "homeScore": 0,
    }


def test_clock_parsing():
    assert pbp.clock_to_seconds("19:32") == 19 * 60 + 32
    assert pbp.clock_to_seconds("0:45") == 45
    assert pbp.clock_to_seconds("0.8") == 0.8
    assert pbp.clock_to_seconds("12.4") == 12.4
    assert pbp.clock_to_seconds("garbage") is None


def test_events_in_clock_window_for_team_and_neighbours():
    events = pbp.parse_events([
        ev("19:32", "Aday Mara makes 3-foot alley oop dunk", team="130", type_="DunkShot", score=2, scoring=True),
        ev("19:14", "Isaiah Evans misses 24-foot three point jumper"),
        ev("19:11", "Patrick Ngongba II Offensive Rebound.", type_="Offensive Rebound"),
        ev("19:10", "Isaiah Evans misses 24-foot three point jumper"),
        ev("19:07", "Aday Mara Defensive Rebound.", team="130", type_="Defensive Rebound"),
        ev("18:59", "Morez Johnson Jr. makes 11-foot pullup jump shot", team="130", score=2, scoring=True),
    ])
    # a Duke possession spanning clock 19:20 -> 19:06
    got = pbp.events_in_window(events, clock_start=19 * 60 + 20, clock_end=19 * 60 + 6, margin=1.0)
    assert [e.clock_text for e in got] == ["19:14", "19:11", "19:10", "19:07"]


def test_outcome_derivation():
    duke = "Duke"
    e = pbp.parse_events([
        ev("19:14", "Isaiah Evans misses 24-foot three point jumper"),
        ev("19:11", "Patrick Ngongba II Offensive Rebound.", type_="Offensive Rebound"),
        ev("19:10", "Isaiah Evans misses 24-foot three point jumper"),
        ev("19:07", "Aday Mara Defensive Rebound.", team="130", type_="Defensive Rebound"),
    ])
    assert pbp.derive_outcome(e, offense=duke) == ("missed_3", 0)
    e = pbp.parse_events([ev("18:59", "Morez Johnson Jr. makes 11-foot pullup jump shot", team="130", score=2, scoring=True)])
    assert pbp.derive_outcome(e, offense="Michigan") == ("made_2", 2)
    e = pbp.parse_events([ev("10:00", "Caleb Foster makes 25-foot three point jumper", score=3, scoring=True)])
    assert pbp.derive_outcome(e, offense=duke) == ("made_3", 3)
    e = pbp.parse_events([
        ev("9:00", "Foul on Elliot Cadeau.", team="130", type_="PersonalFoul"),
        ev("9:00", "Cameron Boozer makes free throw 1 of 2", type_="MadeFreeThrow", score=1, scoring=True),
        ev("9:00", "Cameron Boozer misses free throw 2 of 2", type_="MadeFreeThrow"),
    ])
    assert pbp.derive_outcome(e, offense=duke) == ("free_throws", 1)
    e = pbp.parse_events([ev("8:00", "Dame Sarr Turnover.", type_="Lost Ball Turnover")])
    assert pbp.derive_outcome(e, offense=duke) == ("turnover", 0)
    e = pbp.parse_events([ev("7:00", "Foul on Dame Sarr.", type_="PersonalFoul")])
    assert pbp.derive_outcome(e, offense=duke) == ("foul", 0)
    assert pbp.derive_outcome([], offense=duke) == (None, 0)
    # events by the defence only (e.g. a defensive rebound) give no outcome
    e = pbp.parse_events([ev("7:00", "Aday Mara Defensive Rebound.", team="130", type_="Defensive Rebound")])
    assert pbp.derive_outcome(e, offense=duke) == (None, 0)


def test_team_name_mapping():
    assert pbp.TEAM_BY_ID["150"] == "Duke"
    assert pbp.TEAM_BY_ID["130"] == "Michigan"


def test_assign_events_one_to_one_with_boundary_rules():
    # possession 7 (Michigan) ends at 17:28 where possession 8 (Michigan free throws) starts
    windows = [(7, 1055.0, 1048.0, "Michigan"), (8, 1048.0, 1044.0, "Michigan"), (9, 1044.0, 1034.0, "Duke")]
    events = pbp.parse_events([
        ev("17:28", "Yaxel Lendeborg makes 12-foot pullup jump shot", team="130", score=2, scoring=True),
        ev("17:28", "Foul on Caleb Foster.", team="150", type_="PersonalFoul"),
        ev("17:28", "Aday Mara subbing out for Michigan", team="130", type_="Substitution"),
        ev("17:28", "Yaxel Lendeborg misses free throw 1 of 1", team="130", type_="MadeFreeThrow"),
        ev("17:27", "Cameron Boozer Defensive Rebound.", team="150", type_="Defensive Rebound"),
        ev("17:14", "Foul on Nimari Burnett.", team="130", type_="PersonalFoul"),
    ])
    got = pbp.assign_events(windows, events)
    assert [e.text[:20] for e in got[7]] == ["Yaxel Lendeborg make"]
    assert [e.clock_text for e in got[8]] == ["17:28", "17:28", "17:28", "17:27"]
    assert [e.text[:12] for e in got[9]] == ["Foul on Nima"]
    assert sum(len(v) for v in got.values()) == len(events)


def test_assign_events_gap_after_possession_goes_to_matching_team():
    windows = [(41, 627.0, 616.0, "Duke"), (42, 616.0, 602.0, "Michigan")]
    events = pbp.parse_events([ev("10:12", "Nikolas Khamenia makes layup", score=2, scoring=True)])
    got = pbp.assign_events(windows, events)
    assert len(got[41]) == 1 and got[42] == []
    # possessions without a clock window never receive events
    got = pbp.assign_events([(1, None, None, "Duke")], events)
    assert got == {1: []}


def test_match_player_and_locate_events():
    from basketball_plays.rosters import ROSTERS
    from basketball_plays.scoreboard import ScoreboardRead

    assert pbp.match_player("Morez Johnson Jr. makes layup", ROSTERS) == ("Michigan", "Morez Johnson Jr.")
    assert pbp.match_player("Official TV Timeout", ROSTERS) == (None, None)
    events = pbp.parse_events([ev("19:30", "a"), ev("19:25", "b"), ev("19:00", "c")])
    reads = [ScoreboardRead(t=float(k), clock=1180 - k, clock_text=None, away=0, home=0) for k in range(20)]
    # possession runs video 0..20 s, clock 19:40 (1180) -> 19:20 (1160)
    ts = pbp.locate_events(events, reads, 0.0, 20.0, 1180.0, 1160.0)
    assert ts[0] == 10.0  # read at t=10 shows 1170 = 19:30
    assert ts[1] == 15.0
    assert ts[2] == 20.0  # outside the window: clamped to the end
