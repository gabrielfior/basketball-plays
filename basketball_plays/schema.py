"""Data records shared between the GPU stage (frames.jsonl) and the outputs (trajectories.jsonl)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

# class ids of basketball-player-detection-3-ycjdo/4
CLS_BALL = 0
CLS_BALL_IN_BASKET = 1
CLS_NUMBER = 2
CLS_PLAYER = 3
CLS_PLAYER_IN_POSSESSION = 4
CLS_PLAYER_JUMP_SHOT = 5
CLS_PLAYER_LAYUP_DUNK = 6
CLS_PLAYER_SHOT_BLOCK = 7
CLS_REFEREE = 8
CLS_RIM = 9
PLAYER_CLASSES = (CLS_PLAYER, CLS_PLAYER_IN_POSSESSION, CLS_PLAYER_JUMP_SHOT,
                  CLS_PLAYER_LAYUP_DUNK, CLS_PLAYER_SHOT_BLOCK)


@dataclass
class Detection:
    track_id: int
    cls: int
    conf: float
    bbox: list[float]  # x1, y1, x2, y2 pixels
    team_cluster: int | None = None  # 0/1 from the appearance clustering, None if unknown


@dataclass
class FrameRecord:
    """One sampled video frame as emitted by the GPU stage."""

    frame_idx: int
    t: float  # seconds from the start of the clip
    keypoints: list[list[float]]  # 33 x [x, y, conf]
    detections: list[Detection] = field(default_factory=list)
    ball: list[float] | None = None  # bbox of the best ball detection
    numbers: list[dict] = field(default_factory=list)  # {"track_id": int, "text": str}
    shot: bool = False  # camera shot changed right before this frame

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> FrameRecord:
        dets = [Detection(**x) for x in d.get("detections", [])]
        return cls(
            frame_idx=d["frame_idx"], t=d["t"], keypoints=d["keypoints"], detections=dets,
            ball=d.get("ball"), numbers=d.get("numbers", []), shot=d.get("shot", False),
        )


@dataclass
class PlayerTrack:
    track_id: int
    team: str | None
    jersey: str | None
    name: str | None
    trajectory: list[list[float]]  # [t, x, y] in court feet
    boxes: list[list[float]]  # [t, x1, y1, x2, y2] in pixels


@dataclass
class Possession:
    possession_id: int
    start_time: float
    end_time: float
    fps: float
    offense_team: str | None
    attacking_basket: str  # "left" or "right" in court coordinates
    players: list[PlayerTrack]
    ball: list[list[float]]  # [t, x, y]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> Possession:
        players = [PlayerTrack(**p) for p in d["players"]]
        return cls(**{**d, "players": players})


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_frames(path: str | Path) -> list[FrameRecord]:
    return [FrameRecord.from_dict(d) for d in read_jsonl(path)]


def read_possessions(path: str | Path) -> list[Possession]:
    return [Possession.from_dict(d) for d in read_jsonl(path)]


def write_jsonl(path: str | Path, records) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(r.to_json() + "\n" for r in records)
