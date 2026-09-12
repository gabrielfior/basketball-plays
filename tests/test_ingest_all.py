import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ingest_all", Path(__file__).resolve().parents[1] / "scripts" / "ingest_all.py")
IA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IA)

from basketball_plays.games import Game


def make_game(espn_id, date, opponent, layout, split, home=True):
    return Game(espn_id=espn_id, youtube_id="abcdefghijk", date=date, opponent=opponent,
                home=home, broadcaster="ESPN", layout=layout, split=split)


def test_ordered_puts_layout_first_games_before_the_rest():
    layout_game = make_game(IA.FIRST_PER_LAYOUT["cbs"], "2026-06-01", "B", "cbs", "train")
    games = [
        make_game("1", "2026-01-01", "A", "espn", "train"),
        layout_game,
        make_game("2", "2025-01-01", "C", "espn", "ncaa"),
    ]
    result = IA.ordered(games)
    assert result[0].espn_id == layout_game.espn_id
    assert {g.espn_id for g in result} == {g.espn_id for g in games}


def test_ordered_ranks_train_before_test_before_ncaa_and_dates_ascending():
    train_late = make_game("3", "2026-02-01", "D", "espn", "train")
    train_early = make_game("4", "2026-01-01", "E", "espn", "train")
    test_game = make_game("5", "2026-01-01", "F", "espn", "test")
    ncaa_game = make_game("6", "2026-01-01", "G", "espn", "ncaa")
    result = IA.ordered([train_late, ncaa_game, test_game, train_early])
    assert [g.espn_id for g in result] == ["4", "3", "5", "6"]


def summary_with(home_name, away_name):
    return {
        "header": {"competitions": [{"competitors": [
            {"team": {"id": "1", "displayName": home_name, "name": ""}, "homeAway": "home"},
            {"team": {"id": "2", "displayName": away_name, "name": ""}, "homeAway": "away"},
        ]}]},
        "boxscore": {"players": []},
        "plays": [],
    }


def test_home_check_ok_and_mismatch():
    duke_home = make_game("1", "2026-01-01", "X", "espn", "train", home=True)
    duke_away = make_game("1", "2026-01-01", "X", "espn", "train", home=False)
    summary = summary_with("Duke", "Michigan")
    assert IA.home_check(duke_home, summary) == "ok"
    assert IA.home_check(duke_away, summary) == "mismatch"


def test_home_check_unknown_when_duke_not_in_summary():
    g = make_game("1", "2026-01-01", "X", "espn", "train", home=True)
    summary = summary_with("Michigan", "Kansas")
    assert IA.home_check(g, summary) == "unknown"
