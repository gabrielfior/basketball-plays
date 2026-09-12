from basketball_plays import gameinfo as GI
from basketball_plays import identity
from basketball_plays import playbyplay as pbp


def summary_fixture():
    def team(tid, display, name, abbr):
        return {"id": tid, "displayName": display, "name": name, "abbreviation": abbr,
                "color": "00539b"}
    def athletes(rows):
        return {"statistics": [{"athletes": [
            {"athlete": {"jersey": j, "displayName": n}, "starter": s} for j, n, s in rows]}]}
    return {
        "header": {"competitions": [{"competitors": [
            {"homeAway": "home", "team": team("150", "Duke Blue Devils", "Blue Devils", "DUKE")},
            {"homeAway": "away", "team": team("2599", "St. John's Red Storm", "Red Storm", "SJU")},
        ]}]},
        "boxscore": {"players": [
            {"team": {"id": "2599", "displayName": "St. John's Red Storm"},
             **athletes([("3", "RJ Luis Jr.", True), ("23", "Zuby Ejiofor", False)])},
            {"team": {"id": "150", "displayName": "Duke Blue Devils"},
             **athletes([("12", "Cameron Boozer", True), ("7", "Dame Sarr", False)])},
        ]},
        "plays": [
            {"period": {"number": 1}, "clock": {"displayValue": "19:40"}, "team": {"id": "150"},
             "type": {"text": "JumpShot"}, "text": "Cameron Boozer makes 10-foot jumper",
             "scoringPlay": True, "scoreValue": 2, "awayScore": 0, "homeScore": 2},
            {"period": {"number": 2}, "clock": {"displayValue": "19:40"}, "team": {"id": "2599"},
             "type": {"text": "JumpShot"}, "text": "RJ Luis Jr. misses jumper",
             "scoringPlay": False, "scoreValue": 0, "awayScore": 0, "homeScore": 2},
            {"period": {"number": 3}, "clock": {"displayValue": "4:59"}, "team": {"id": "150"},
             "type": {"text": "JumpShot"}, "text": "Dame Sarr makes layup",
             "scoringPlay": True, "scoreValue": 2, "awayScore": 60, "homeScore": 62},
        ],
    }


def test_from_summary_reads_teams_rosters_starters_and_periods():
    info = GI.from_summary(summary_fixture())
    assert (info.home, info.away) == ("Duke", "St. John's")
    assert info.team_by_id == {"150": "Duke", "2599": "St. John's"}
    assert info.rosters["Duke"] == {"12": "Cameron Boozer", "7": "Dame Sarr"}
    assert info.rosters["St. John's"]["3"] == "RJ Luis Jr."
    assert info.starters["Duke"] == ["Cameron Boozer"]
    assert info.periods == [1, 2, 3]


def test_short_name_strips_the_mascot_only():
    assert GI.short_name("North Carolina Tar Heels", "Tar Heels") == "North Carolina"
    assert GI.short_name("St. John's Red Storm", "Red Storm") == "St. John's"
    assert GI.short_name("Duke Blue Devils", "Blue Devils") == "Duke"


def test_parse_events_uses_the_game_team_map():
    info = GI.from_summary(summary_fixture())
    ev = pbp.parse_events(summary_fixture()["plays"], period=2, team_by_id=info.team_by_id)
    assert [e.team for e in ev] == ["St. John's"]
    assert pbp.parse_events(summary_fixture()["plays"], period=3, team_by_id=info.team_by_id)[0].clock == 299


def test_period_length():
    assert pbp.period_length(1) == 1200 and pbp.period_length(2) == 1200
    assert pbp.period_length(3) == 300 and pbp.period_length(4) == 300


def test_name_clusters_with_game_rosters():
    info = GI.from_summary(summary_fixture())
    reads = {0: ["3", "23", "3"], 1: ["12", "7", "12"]}
    out = identity.name_clusters({0: 0.9, 1: 0.2}, reads, home_team=info.home, away_team=info.away,
                                 rosters=info.rosters)
    assert out == {0: "St. John's", 1: "Duke"}
