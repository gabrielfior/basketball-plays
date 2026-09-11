"""ESPN play-by-play: parsing, clock windows, and possession outcome derivation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from basketball_plays.rosters import DUKE, MICHIGAN

GAME_ID = "401817238"
SUMMARY_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/"
               "summary?event={game_id}")
TEAM_BY_ID = {"150": DUKE, "130": MICHIGAN}


@dataclass
class Event:
    clock_text: str
    clock: float  # seconds remaining in the period
    team: str | None
    type: str
    text: str
    scoring: bool
    score_value: int
    away_score: int
    home_score: int

    def to_dict(self) -> dict:
        return asdict(self)


def clock_to_seconds(text: str) -> float | None:
    text = text.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})\.(\d)", text)
    if m:
        return float(text)
    return None


def fetch_summary(game_id: str = GAME_ID, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        return json.loads(cache.read_text())
    data = requests.get(SUMMARY_URL.format(game_id=game_id), timeout=30).json()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data


def parse_events(plays: list[dict], period: int = 1) -> list[Event]:
    out = []
    for p in plays:
        if p.get("period", {}).get("number") != period:
            continue
        clock_text = p.get("clock", {}).get("displayValue", "")
        clock = clock_to_seconds(clock_text)
        if clock is None:
            continue
        out.append(Event(
            clock_text=clock_text, clock=clock, team=TEAM_BY_ID.get(str(p.get("team", {}).get("id"))),
            type=p.get("type", {}).get("text", ""), text=p.get("text", ""),
            scoring=bool(p.get("scoringPlay")), score_value=int(p.get("scoreValue") or 0),
            away_score=int(p.get("awayScore") or 0), home_score=int(p.get("homeScore") or 0),
        ))
    return out


def events_in_window(events: list[Event], clock_start: float, clock_end: float, margin: float = 1.0) -> list[Event]:
    """Events whose clock lies within a possession's clock range (start > end; clock counts down)."""
    hi, lo = max(clock_start, clock_end) + margin, min(clock_start, clock_end) - margin
    return [e for e in events if lo <= e.clock <= hi]


def derive_outcome(events: list[Event], offense: str | None) -> tuple[str | None, int]:
    """Summarise the offense's events into (outcome, points).

    Outcomes: made_2, made_3, missed_2, missed_3, free_throws, turnover, foul, or None.
    The last shot attempt by the offense decides; free throws and turnovers take precedence over
    a preceding miss when they come later in the sequence.
    """
    own = [e for e in events if e.team == offense] if offense else list(events)
    points = sum(e.score_value for e in own if e.scoring)
    outcome = None
    for e in own:
        text = e.text.lower()
        if "free throw" in text:
            outcome = "free_throws"
        elif "turnover" in e.type.lower() or "turnover" in text:
            outcome = "turnover"
        elif " makes " in text or " misses " in text:
            made = " makes " in text
            three = "three point" in text
            outcome = f"{'made' if made else 'missed'}_{3 if three else 2}"
        elif "foul" in e.type.lower() and outcome is None:
            outcome = "foul"
    return outcome, points


DEAD_BALL_TYPES = ("foul", "free throw", "substitution", "timeout", "subbing", "challenge", "jumpball")


def _offensive(e: Event) -> bool:
    """Events only the team with the ball can produce (they end or extend its possession)."""
    text = e.text.lower()
    return (" makes " in text or " misses " in text) and "free throw" not in text or "turnover" in text \
        or "turnover" in e.type.lower() or "offensive rebound" in text


def _dead_ball(e: Event) -> bool:
    text = (e.type + " " + e.text).lower()
    return any(k in text for k in DEAD_BALL_TYPES)


def assign_events(windows: list[tuple[int, float | None, float | None, str | None]], events: list[Event],
                  margin: float = 1.0, gap_slack: float = 12.0) -> dict[int, list[Event]]:
    """Assign each event to exactly one possession.

    `windows` holds (possession_id, clock_start, clock_end, offense_team) in video order. Rules:
    a team's event goes to a candidate possession of that team when one exists; an event sitting
    on the shared boundary of two possessions goes to the earlier one if it ends a possession
    (shot, turnover, rebound) and to the later one if it is dead-ball administration (foul, free
    throw, substitution, timeout); an offensive event (shot, turnover, offensive rebound) that no
    window of its team contains goes to that team's most recent possession if it ended within
    `gap_slack` clock seconds before it; defensive events stay with the containing window.
    """
    known = [(pid, cs, ce, team) for pid, cs, ce, team in windows if cs is not None and ce is not None]
    out: dict[int, list[Event]] = {pid: [] for pid, *_ in windows}
    for e in events:
        cands = [w for w in known if min(w[1], w[2]) - margin <= e.clock <= max(w[1], w[2]) + margin]
        same_team = [w for w in cands if w[3] == e.team]
        if e.team is not None and same_team:
            cands = same_team
        elif e.team is not None and _offensive(e):
            # an offensive event with no window of that team around it: the team's most recent
            # possession that ended shortly before it (the play ran on during a replay) takes it
            recent = [w for w in known if w[3] == e.team and w[1] >= e.clock and min(w[1], w[2]) - e.clock <= gap_slack]
            if recent:
                out[min(recent, key=lambda w: min(w[1], w[2]) - e.clock)[0]].append(e)
                continue
        if not cands:
            continue
        strict = [w for w in cands if min(w[1], w[2]) < e.clock < max(w[1], w[2])]
        if len(strict) == 1:
            out[strict[0][0]].append(e)
            continue
        pick = cands[-1] if _dead_ball(e) else cands[0]
        out[pick[0]].append(e)
    return out
