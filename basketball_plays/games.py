"""Game registry (games.json) and the per-game data directory layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

LAYOUTS = ("espn", "cbs", "cw", "ncaa", "cbssn")
SPLITS = ("train", "test", "ncaa")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "games.json"
DEFAULT_ROOT = Path("data/games")


@dataclass(frozen=True)
class Game:
    espn_id: str
    youtube_id: str
    date: str
    opponent: str
    home: bool
    broadcaster: str
    layout: str
    split: str
    note: str = ""

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.youtube_id}"


@dataclass(frozen=True)
class GamePaths:
    root: Path
    video: Path
    raw_dir: Path
    trajectories: Path
    scoreboard_raw: Path
    espn_summary: Path
    halfcourt: Path
    coverage: Path
    game_json: Path = field(default=Path("game.json"))

    @classmethod
    def for_game(cls, espn_id: str, root: str | Path = DEFAULT_ROOT) -> GamePaths:
        d = Path(root) / espn_id
        return cls(root=d, video=d / "video.mp4", raw_dir=d / "raw",
                   trajectories=d / "trajectories.jsonl", scoreboard_raw=d / "scoreboard_raw.jsonl",
                   espn_summary=d / "espn_summary.json", halfcourt=d / "halfcourt.jsonl",
                   coverage=d / "coverage.json", game_json=d / "game.json")


def _validate(g: Game) -> None:
    if g.layout not in LAYOUTS:
        raise ValueError(f"{g.espn_id}: unknown layout {g.layout!r}")
    if g.split not in SPLITS:
        raise ValueError(f"{g.espn_id}: unknown split {g.split!r}")
    try:
        date.fromisoformat(g.date)
    except ValueError as e:
        raise ValueError(f"{g.espn_id}: bad date {g.date!r}") from e
    if len(g.youtube_id) != 11:
        raise ValueError(f"{g.espn_id}: bad youtube id {g.youtube_id!r}")


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> list[Game]:
    entries = json.loads(Path(path).read_text())
    games = [Game(**e) for e in entries]
    seen: set[str] = set()
    for g in games:
        _validate(g)
        if g.espn_id in seen:
            raise ValueError(f"duplicate espn_id {g.espn_id}")
        seen.add(g.espn_id)
    return games


def get_game(espn_id: str, path: str | Path = DEFAULT_REGISTRY) -> Game:
    for g in load_registry(path):
        if g.espn_id == espn_id:
            return g
    raise KeyError(espn_id)


def game_dir(espn_id: str, root: str | Path = DEFAULT_ROOT) -> Path:
    return Path(root) / espn_id
