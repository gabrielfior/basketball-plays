import json

import pytest

from basketball_plays import games as G


def write_registry(tmp_path, entries):
    p = tmp_path / "games.json"
    p.write_text(json.dumps(entries))
    return p


GOOD = {"espn_id": "401817238", "youtube_id": "gSYUO3E3MBY", "date": "2026-02-21",
        "opponent": "Michigan", "home": True, "broadcaster": "ESPN", "layout": "espn",
        "split": "train"}


def test_load_registry_parses_entries(tmp_path):
    p = write_registry(tmp_path, [GOOD])
    games = G.load_registry(p)
    assert len(games) == 1
    g = games[0]
    assert (g.espn_id, g.youtube_id, g.layout, g.split, g.home) == (
        "401817238", "gSYUO3E3MBY", "espn", "train", True)
    assert g.note == ""


@pytest.mark.parametrize("bad", [
    {**GOOD, "layout": "fox"},
    {**GOOD, "split": "validation"},
    {**GOOD, "date": "21/02/2026"},
    {**GOOD, "youtube_id": "short"},
])
def test_load_registry_rejects_bad_entries(tmp_path, bad):
    with pytest.raises(ValueError):
        G.load_registry(write_registry(tmp_path, [bad]))


def test_load_registry_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError):
        G.load_registry(write_registry(tmp_path, [GOOD, GOOD]))


def test_game_paths_live_under_the_game_dir(tmp_path):
    paths = G.GamePaths.for_game("401817238", root=tmp_path)
    assert paths.video == tmp_path / "401817238" / "video.mp4"
    assert paths.raw_dir == tmp_path / "401817238" / "raw"
    assert paths.trajectories.name == "trajectories.jsonl"
    assert paths.halfcourt.name == "halfcourt.jsonl"
    assert paths.coverage.name == "coverage.json"


def test_repo_registry_is_valid_and_complete():
    games = G.load_registry()
    assert len(games) == 27
    assert sum(g.split == "test" for g in games) == 6
    assert sum(g.split == "ncaa" for g in games) == 4
    assert {g.layout for g in games} <= set(G.LAYOUTS)
    assert any(g.espn_id == "401817238" and g.youtube_id == "gSYUO3E3MBY" for g in games)
