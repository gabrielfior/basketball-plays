"""Duke half-court possession records: ESPN-defined intervals, clock-to-video mapping, track
gathering, t0 and setup detection. See the 2026-09-11 design spec, Definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from basketball_plays.playbyplay import Event

ATO, DEAD, LIVE, PERIOD = "ato", "dead", "live", "period"
SHOT, TURNOVER, FOUL, STOPPAGE, PERIOD_END = "shot", "turnover", "foul", "stoppage", "period_end"


@dataclass
class Interval:
    team: str
    start_clock: float
    end_clock: float
    start_type: str
    terminal: str
    events: list[Event] = field(default_factory=list)
    free_throws: bool = False


def _low(e: Event) -> str:
    return (e.type + " " + e.text).lower()


def _is_shot(e: Event) -> bool:
    t = e.text.lower()
    return (" makes " in t or " misses " in t) and "free throw" not in t


def _is_made(e: Event) -> bool:
    return " makes " in e.text.lower()


def _is_free_throw(e: Event) -> bool:
    return "free throw" in e.text.lower()


def _is_last_free_throw(e: Event) -> bool:
    m = re.search(r"free throw (\d) of (\d)", e.text.lower())
    return bool(m) and m.group(1) == m.group(2)


def _is_turnover(e: Event) -> bool:
    return "turnover" in _low(e)


def _is_steal(e: Event) -> bool:
    return "steal" in e.type.lower()


def _is_foul(e: Event) -> bool:
    return "foul" in e.type.lower()


def _is_timeout(e: Event) -> bool:
    return "timeout" in e.type.lower()


def _is_def_rebound(e: Event) -> bool:
    return e.type.lower().startswith("defensive rebound")


def _is_off_rebound(e: Event) -> bool:
    return e.type.lower().startswith("offensive rebound")


def _is_dead_rebound(e: Event) -> bool:
    return e.type.lower().startswith("dead ball rebound")


def _is_jumpball(e: Event) -> bool:
    return "jumpball" in e.type.lower()


def _is_substitution(e: Event) -> bool:
    return "substitution" in e.type.lower() or "subbing" in e.text.lower()


def _is_period_end(e: Event) -> bool:
    return e.type.lower() == "end period"


def _free_throws_follow(events: list[Event], i: int, team: str) -> bool:
    """True when the next non-substitution events at the same clock are `team` free throws."""
    clock = events[i].clock
    for e in events[i + 1:]:
        if e.clock != clock:
            return False
        if _is_substitution(e) or _is_timeout(e):
            continue
        return _is_free_throw(e) and e.team == team
    return False


def intervals(events: list[Event], period_length: float = 1200.0) -> list[Interval]:
    """Split one period's events into possession intervals per team with a start type.

    Boundaries: made shots, defensive rebounds, turnovers, made last free throws (change of
    possession); fouls and timeouts (stoppage, same team keeps the ball with a fresh setup);
    shooting fouls close the fouled team's interval with `terminal="foul"` and the free throws
    are not part of any interval. A timeout marks the next interval `ato`. Intervals with no
    clock elapsed are dropped.
    """
    out: list[Interval] = []
    teams = sorted({e.team for e in events if e.team})
    other = {teams[0]: teams[-1], teams[-1]: teams[0]} if len(teams) == 2 else {}
    cur: Interval | None = None
    pending_ato = False
    last_shot: Event | None = None
    technical_holder: str | None = None
    technical_pending = False
    technical_clock: float | None = None

    def close(end_clock: float, terminal: str, free_throws: bool = False) -> None:
        nonlocal cur
        if cur is not None:
            cur.end_clock, cur.terminal, cur.free_throws = end_clock, terminal, free_throws
            out.append(cur)
            cur = None

    def open_(team: str | None, clock: float, start_type: str) -> None:
        nonlocal cur, pending_ato, last_shot
        if team is None:
            return
        cur = Interval(team, clock, clock, ATO if pending_ato else start_type, STOPPAGE)
        pending_ato = False
        last_shot = None

    for i, e in enumerate(events):
        if _is_substitution(e):
            continue
        if technical_pending and not _is_free_throw(e):
            # the technical's free throws never came (or already ran out): the team that had
            # the ball keeps it, on a fresh dead-ball setup
            open_(technical_holder, technical_clock, DEAD)
            technical_pending, technical_holder = False, None
        if _is_period_end(e):
            close(0.0, PERIOD_END)
            break
        if _is_jumpball(e):
            if "won by" in e.text.lower():
                close(e.clock, STOPPAGE)
                open_(e.team, e.clock, LIVE)
            continue
        if _is_timeout(e):
            team = cur.team if cur else None
            close(e.clock, STOPPAGE)
            pending_ato = True
            open_(team, e.clock, DEAD)
            continue
        if _is_free_throw(e):
            if technical_pending:
                # possession does not change on a technical foul; ignore who shoots it
                if _is_last_free_throw(e) or "1 of 1" in e.text.lower():
                    open_(technical_holder, e.clock, DEAD)
                    technical_pending, technical_holder = False, None
                continue
            # free throws belong to no interval; an interval opened at this clock for the
            # fouling team (and-one) is discarded by the zero-duration filter
            if cur is not None and cur.team != e.team:
                cur = None
            if (_is_last_free_throw(e) or "1 of 1" in e.text.lower()) and _is_made(e):
                open_(other.get(e.team), e.clock, DEAD)
            continue
        if cur is not None:
            cur.events.append(e)
        if _is_shot(e):
            if cur is None or cur.team != e.team:
                close(e.clock, STOPPAGE)
                if not out:
                    # the period's first shot with no jump ball on record: start of period
                    open_(e.team, period_length, PERIOD)
                else:
                    # possession bookkeeping got out of sync; start a fresh interval for the
                    # shooter
                    open_(e.team, e.clock, LIVE)
                cur.events.append(e)
            last_shot = e
            if _is_made(e):
                close(e.clock, SHOT)
                open_(other.get(e.team), e.clock, DEAD)
            continue
        if _is_off_rebound(e):
            if cur is None:
                open_(e.team, e.clock, LIVE)
            continue
        if _is_def_rebound(e):
            end = last_shot.clock if last_shot is not None and cur is not None else e.clock
            close(end, SHOT)
            open_(e.team, e.clock, LIVE)
            continue
        if _is_dead_rebound(e):
            end = last_shot.clock if last_shot is not None and cur is not None else e.clock
            close(end, SHOT)
            open_(e.team, e.clock, DEAD)
            continue
        if _is_turnover(e):
            nxt = events[i + 1] if i + 1 < len(events) else None
            stolen = nxt is not None and _is_steal(nxt) and nxt.clock == e.clock
            close(e.clock, TURNOVER)
            open_(other.get(e.team), e.clock, LIVE if stolen else DEAD)
            continue
        if _is_steal(e):
            continue
        if _is_foul(e):
            if "technical" in _low(e):
                # possession doesn't change; remember who had it until the technical FTs resolve
                holder = cur.team if cur is not None else None
                close(e.clock, STOPPAGE)
                technical_holder, technical_pending, technical_clock = holder, True, e.clock
                continue
            fouled = other.get(e.team)
            ft_follow = _free_throws_follow(events, i, fouled)
            if cur is not None and cur.team == e.team and not ft_follow:
                # offensive foul: a turnover
                close(e.clock, TURNOVER)
                open_(fouled, e.clock, DEAD)
            elif fouled is not None and ft_follow:
                if cur is not None and cur.team == fouled:
                    close(e.clock, FOUL, free_throws=True)
                else:
                    cur = None  # and-one on a made shot: the fresh interval never happened
            else:
                team = cur.team if cur else fouled
                close(e.clock, STOPPAGE)
                open_(team, e.clock, DEAD)
            continue
    # an interval still open here means the event list was truncated ("End Period" closes a real
    # period inside the loop); it is dropped rather than guessed
    return [iv for iv in out if iv.end_clock < iv.start_clock]
