"""Drawing helpers for the court animation and the video overlay."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from basketball_plays.court import NCAA, draw_court, to_pixel
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import Possession

TEAM_BGR = {DUKE: (155, 83, 0), MICHIGAN: (5, 203, 255), None: (128, 128, 128)}
BALL_BGR = (0, 110, 255)
TEXT_BGR = (30, 30, 30)


def finite(row) -> bool:
    return all(np.isfinite(v) for v in row)


def team_color(team: str | None) -> tuple[int, int, int]:
    return TEAM_BGR.get(team, TEAM_BGR[None])


SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}


def surname(name: str) -> str:
    parts = [p for p in name.split() if p.lower() not in SUFFIXES]
    return parts[-1] if parts else name


def label_for(team: str | None, jersey: str | None, name: str | None, track_id: int) -> str:
    short = (team or "?")[:3].upper()
    if name:
        return f"{short} #{jersey} {surname(name)}"
    if jersey:
        return f"{short} #{jersey}"
    return f"{short} id{track_id}"


def clock_text(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    return f"{seconds:.1f}"


def _index_by_time(rows: list[list[float]]) -> dict[float, list[float]]:
    return {round(r[0], 3): r[1:] for r in rows}


def possession_times(pos: Possession) -> list[float]:
    n = int(round((pos.end_time - pos.start_time) * pos.fps))
    return [round(pos.start_time + k / pos.fps, 3) for k in range(n)]


EVENT_HOLD_S = 3.0
GREEN, RED, AMBER, ORANGE, GREY = (60, 170, 60), (40, 40, 220), (0, 200, 255), (0, 120, 255), (120, 120, 120)


def event_style(e: dict) -> tuple[str, tuple[int, int, int]] | None:
    """(tag, colour) for an event banner; None for events not worth showing (substitutions)."""
    text = e.get("text", "").lower()
    etype = e.get("type", "").lower()
    if "subbing" in text or "substitution" in etype:
        return None
    if "free throw" in text:
        return ("FT MADE", GREEN) if " makes " in text else ("FT MISSED", RED)
    if " makes " in text:
        return (f"MADE +{e.get('score_value') or (3 if 'three point' in text else 2)}", GREEN)
    if " misses " in text:
        return ("MISSED 3" if "three point" in text else "MISSED 2", RED)
    if "turnover" in text or "turnover" in etype:
        return ("TURNOVER", ORANGE)
    if "steal" in etype or "steal" in text:
        return ("STEAL", ORANGE)
    if "block" in etype:
        return ("BLOCK", ORANGE)
    if "foul" in etype or "foul" in text:
        return ("FOUL", AMBER)
    if "rebound" in etype:
        return ("OFF REB" if "offensive" in etype else "DEF REB", GREY)
    if "timeout" in etype:
        return ("TIMEOUT", GREY)
    return (e.get("type", "EVENT").upper(), GREY)


def active_events(pos: Possession, t: float, hold: float = EVENT_HOLD_S) -> list[dict]:
    """Events whose video time has passed within the last `hold` seconds, latest first."""
    evs = [e for e in (pos.events or []) if e.get("t") is not None and e["t"] <= t + 1e-6 < e["t"] + hold]
    return sorted(evs, key=lambda e: -e["t"])


def draw_banner(img: np.ndarray, x: int, y: int, tag: str, text: str, color, scale: float = 0.5) -> int:
    """Draw a coloured tag box followed by the event text; returns the banner height."""
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    (ew, eh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    h = max(th, eh) + 12
    cv2.rectangle(img, (x, y), (x + tw + 12, y + h), color, -1)
    cv2.rectangle(img, (x + tw + 12, y), (x + tw + 12 + ew + 12, y + h), (250, 250, 250), -1)
    cv2.putText(img, tag, (x + 6, y + h - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + tw + 18, y + h - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_BGR, 1, cv2.LINE_AA)
    return h


def player_for_event(pos: Possession, e: dict):
    name = e.get("player")
    if not name:
        return None
    for p in pos.players:
        if p.name == name:
            return p
    return None


def position_at(p, t: float):
    for row in p.trajectory:
        if abs(row[0] - t) < 0.051 and finite(row):
            return row[1], row[2]
    return None


def render_court_frame(
    pos: Possession, t: float, scale: float = 10.0, padding: int = 30, trail_seconds: float = 2.0,
    header: int = 44, court_img: np.ndarray | None = None,
) -> np.ndarray:
    court = (court_img if court_img is not None else draw_court(NCAA, scale, padding)).copy()
    for p in pos.players:
        pts = [r for r in p.trajectory if t - trail_seconds <= r[0] <= t + 1e-6 and finite(r)]
        if not pts:
            continue
        color = team_color(p.team)
        pix = [to_pixel((x, y), scale, padding) for _, x, y in pts]
        for k in range(1, len(pix)):
            alpha = k / len(pix)
            c = tuple(int(v * alpha + 255 * (1 - alpha) * 0.3) for v in color)
            cv2.line(court, pix[k - 1], pix[k], c, 2, cv2.LINE_AA)
        cx, cy = pix[-1]
        cv2.circle(court, (cx, cy), int(1.2 * scale), color, -1, cv2.LINE_AA)
        cv2.circle(court, (cx, cy), int(1.2 * scale), (255, 255, 255), 1, cv2.LINE_AA)
        txt = p.jersey or str(p.track_id % 1000)
        cv2.putText(court, txt, (cx - 6 * len(txt), cy - int(1.6 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, TEXT_BGR, 1, cv2.LINE_AA)
    ball = [r for r in pos.ball if abs(r[0] - t) < 0.5 / pos.fps and finite(r)]
    if ball:
        bx, by = to_pixel((ball[0][1], ball[0][2]), scale, padding)
        cv2.circle(court, (bx, by), int(0.8 * scale), BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(court, (bx, by), int(0.8 * scale), (0, 0, 0), 1, cv2.LINE_AA)
    # event banners and a ring around the player named in the event
    y = court.shape[0] - 12
    for e in active_events(pos, t):
        style = event_style(e)
        if style is None:
            continue
        tag, color = style
        h = draw_banner(court, 12, y - 30, tag, f"{e.get('clock_text', '')}  {e.get('text', '')}"[:90], color)
        y -= h + 6
        player = player_for_event(pos, e)
        xy = position_at(player, t) if player else None
        if xy:
            cx, cy = to_pixel(xy, scale, padding)
            cv2.circle(court, (cx, cy), int(2.2 * scale), color, 3, cv2.LINE_AA)
    bar = np.full((header, court.shape[1], 3), (245, 245, 245), dtype=np.uint8)
    off = pos.offense_team or "?"
    txt = (f"Possession {pos.possession_id}  |  {off} offense -> {pos.attacking_basket} basket  |  "
           f"video {pos.start_time:7.1f}s - {pos.end_time:7.1f}s  |  t={t:7.1f}s")
    if pos.clock_start is not None:
        txt += f"  |  clock {clock_text(pos.clock_start)}"
    if pos.outcome:
        txt += f"  |  {pos.outcome.replace('_', ' ')}" + (f" (+{pos.points_scored})" if pos.points_scored else "")
    cv2.putText(bar, txt, (12, header - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_BGR, 1, cv2.LINE_AA)
    return np.vstack([bar, court])


def draw_overlay(frame: np.ndarray, pos: Possession, t: float, box_index: dict[int, dict]) -> np.ndarray:
    out = frame
    y = 16
    highlighted = {}
    for e in active_events(pos, t):
        style = event_style(e)
        if style is None:
            continue
        tag, color = style
        y += draw_banner(out, 16, y, tag, f"{e.get('clock_text', '')}  {e.get('text', '')}"[:80], color, scale=0.6) + 6
        player = player_for_event(pos, e)
        if player is not None:
            highlighted[player.track_id] = color
    for p in pos.players:
        hl = highlighted.get(p.track_id)
        if hl is not None:
            box = box_index[p.track_id].get(t)
            if box is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(out, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), hl, 4)
    for p in pos.players:
        box = box_index[p.track_id].get(t)
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = team_color(p.team)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = label_for(p.team, p.jersey, p.name, p.track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return out


def clip_name(pos: Possession) -> str:
    off = (pos.offense_team or "unknown").lower()
    return f"p{pos.possession_id:03d}_{int(pos.start_time):04d}s_{off}_{pos.outcome or 'none'}.mp4"


def box_lookup(pos: Possession) -> dict[int, dict[float, list[float]]]:
    return {p.track_id: _index_by_time(p.boxes) for p in pos.players}


class VideoWriter:
    """cv2 writer with an ffmpeg h264 re-encode on close for broad playback compatibility."""

    def __init__(self, path: str | Path, fps: float, size_wh: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(".raw.mp4")
        self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size_wh)
        self.size = size_wh
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)
        self.n += 1

    def close(self) -> None:
        self.writer.release()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "23", str(self.path)], check=True,
            )
            self.tmp.unlink()
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.tmp.rename(self.path)
