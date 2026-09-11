"""Duke half-court possession records: ESPN-defined intervals, clock-to-video mapping, track
gathering, t0 and setup detection. See the 2026-09-11 design spec, Definitions."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from basketball_plays import zones
from basketball_plays.playbyplay import Event, derive_outcome
from basketball_plays.schema import Possession

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


def clock_to_video(reads, clock: float, mode: str,
                   max_gap_s: float = 20.0) -> float | None:
    """Video time at which the scoreboard showed `clock`.

    `mode="first"` is the moment the clock reached the value (a running-clock event such as a
    rebound or shot); `mode="last"` is the moment just before it moved on (the inbound after a
    stoppage). Falls back to linear interpolation between the nearest reads on either side, or
    None when that pair is more than `max_gap_s` seconds of video apart.
    """
    if mode not in ("first", "last"):
        raise ValueError(mode)
    known = sorted(((r.t, r.clock) for r in reads if r.clock is not None),
                   key=lambda p: p[0])
    exact = [t for t, c in known if abs(c - clock) <= 0.5]
    if exact:
        return float(exact[0] if mode == "first" else exact[-1])
    above = [(t, c) for t, c in known if c > clock]
    below = [(t, c) for t, c in known if c < clock]
    if not above or not below:
        return None
    t_hi, c_hi = max(above, key=lambda p: p[0])  # latest read still above the target clock
    t_lo, c_lo = min(below, key=lambda p: p[0])  # earliest read already below it
    if t_lo <= t_hi or t_lo - t_hi > max_gap_s:
        return None
    frac = (c_hi - clock) / (c_hi - c_lo)
    return round(float(t_hi + frac * (t_lo - t_hi)), 2)


MIN_STILL_PLAYERS = 4
ARC_FT = 22.0
TRANSITION_S = 6.0
DEAD_SEARCH = (-3.0, 1.5)
LIVE_SEARCH_S = 6.0
MIN_CENTROID_PLAYERS = 3


@dataclass
class Track:
    track_id: int
    name: str | None
    jersey: str | None
    xy: dict[float, tuple[float, float]]  # rounded video time -> canonical court position
    holding: set[float]


def gather_tracks(possessions: list[Possession], team: str, t_start: float,
                   t_end: float) -> list[Track]:
    """Every `team` track from any vision possession overlapping [t_start, t_end], clipped to
    the window, mirrored so the offence attacks x = 5.25. NaN positions are dropped."""
    out: list[Track] = []
    for pos in possessions:
        if pos.end_time < t_start or pos.start_time > t_end:
            continue
        for p in pos.players:
            if p.team != team:
                continue
            rows = [r for r in p.trajectory
                    if t_start <= r[0] <= t_end and np.isfinite(r[1]) and np.isfinite(r[2])]
            if not rows:
                continue
            xy = zones.mirror_to_canonical(np.array([[r[1], r[2]] for r in rows]),
                                            pos.attacking_basket)
            out.append(Track(
                track_id=p.track_id, name=p.name, jersey=p.jersey,
                xy={round(r[0], 3): (float(x), float(y)) for r, (x, y) in zip(rows, xy)},
                holding={round(h, 3) for h in p.holding if t_start <= h <= t_end},
            ))
    return out


def ball_handler_series(tracks: list[Track]) -> dict[float, tuple[float, float]]:
    out: dict[float, tuple[float, float]] = {}
    for tr in tracks:
        for t in tr.holding:
            if t in tr.xy and t not in out:
                out[t] = tr.xy[t]
    return dict(sorted(out.items()))


def _times(tracks: list[Track]) -> list[float]:
    return sorted({t for tr in tracks for t in tr.xy})


def positions_at(tracks: list[Track], t: float,
                  max_move_ft: float = 1.0) -> list[tuple[float, float]]:
    """Every track's position at rounded time `t`, id-agnostic: a second track id sitting
    within `max_move_ft` of an already-kept position (the same player, fragmented into two
    track ids) is treated as a duplicate and dropped rather than counted twice."""
    t = round(t, 3)
    out: list[tuple[float, float]] = []
    for tr in tracks:
        p = tr.xy.get(t)
        if p is None:
            continue
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < max_move_ft for q in out):
            continue
        out.append(p)
    return out


def find_t0(tracks: list[Track], handler: dict[float, tuple[float, float]], t_start: float,
            t_end: float, fps: float = 10.0) -> float | None:
    """First time the ball handler is in the frontcourt; else the first time at least three
    visible players have their median x in the frontcourt."""
    for t, (x, _) in sorted(handler.items()):
        if t_start <= t <= t_end and x < zones.HALF_COURT_X:
            return t
    for t in _times(tracks):
        if not t_start <= t <= t_end:
            continue
        positions = positions_at(tracks, t)
        if len(positions) < MIN_CENTROID_PLAYERS:
            continue
        if float(np.median([p[0] for p in positions])) < zones.HALF_COURT_X:
            return t
    return None


def still_players(tracks: list[Track], t: float, window_s: float = 0.5,
                   max_move_ft: float = 1.0) -> tuple[int, int]:
    """(visible, still): id-agnostic player positions at t, and how many of them also have a
    position `window_s` earlier within `max_move_ft`, regardless of which track id it came
    from (a track break moves the earlier position to a different Track)."""
    now = positions_at(tracks, t, max_move_ft)
    prev = positions_at(tracks, t - window_s, max_move_ft)
    still = sum(
        1 for (x1, y1) in now
        if any(np.hypot(x1 - x0, y1 - y0) < max_move_ft for (x0, y0) in prev)
    )
    return len(now), still


def _is_still_frame(tracks: list[Track], t: float) -> bool:
    visible, still = still_players(tracks, t)
    return visible >= MIN_STILL_PLAYERS and still >= MIN_STILL_PLAYERS


def find_setup(tracks: list[Track], handler: dict[float, tuple[float, float]], start_type: str,
               t_start: float, t0: float, t_end: float, fps: float = 10.0) -> tuple[float, bool]:
    """(setup_time, no_setup) per the spec's setup-frame rule."""
    times = _times(tracks)
    if start_type in (DEAD, ATO):
        lo, hi = t_start + DEAD_SEARCH[0], t_start + DEAD_SEARCH[1]
        cands = [t for t in times if lo <= t <= hi and _is_still_frame(tracks, t)]
        if cands:
            return cands[-1], False
    hi = min(t0 + LIVE_SEARCH_S, t_end)
    for t in times:
        if not t0 <= t <= hi or not _is_still_frame(tracks, t):
            continue
        h = handler.get(round(t, 3))
        if h is not None and float(zones.rim_distance(np.array([h]))[0]) <= ARC_FT:
            continue
        return t, False
    return t0, True


@dataclass
class HalfcourtRecord:
    game_id: str
    index: int
    team: str
    start_type: str
    terminal: str
    free_throws: bool
    clock_start: float
    clock_end: float
    t_start: float | None
    t_end: float | None
    t0: float | None
    setup: float | None
    no_setup: bool
    transition: bool
    located: bool
    outcome: str | None
    points: int
    n_visible_at_setup: int
    players: list[dict] = field(default_factory=list)
    ball_handler: list[list[float]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> HalfcourtRecord:
        return cls(**d)


def _unlocated(game_id: str, index: int, iv: Interval, outcome: str | None,
               points: int) -> HalfcourtRecord:
    return HalfcourtRecord(
        game_id=game_id, index=index, team=iv.team, start_type=iv.start_type, terminal=iv.terminal,
        free_throws=iv.free_throws, clock_start=iv.start_clock, clock_end=iv.end_clock,
        t_start=None, t_end=None, t0=None, setup=None, no_setup=True, transition=False,
        located=False, outcome=outcome, points=points, n_visible_at_setup=0,
        events=[e.to_dict() for e in iv.events],
    )


def build_records(game_id: str, team: str, events: list[Event], reads,
                   possessions: list[Possession],
                   period_length: float = 1200.0) -> list[HalfcourtRecord]:
    out: list[HalfcourtRecord] = []
    for index, iv in enumerate(i for i in intervals(events, period_length) if i.team == team):
        outcome, points = derive_outcome(iv.events, team)
        start_mode = "first" if iv.start_type == LIVE else "last"
        t_start = clock_to_video(reads, iv.start_clock, start_mode)
        t_end = clock_to_video(reads, iv.end_clock, "first")
        if t_start is None or t_end is None or t_end <= t_start:
            out.append(_unlocated(game_id, index, iv, outcome, points))
            continue
        t_end = round(t_end + 0.5, 2)  # the read at the terminal clock precedes the event by 1 s
        tracks = gather_tracks(possessions, team, t_start - 3.0, t_end)
        handler = ball_handler_series(tracks)
        t0 = find_t0(tracks, handler, t_start - 1.0, t_end)
        if t0 is None:
            rec = _unlocated(game_id, index, iv, outcome, points)
            rec.t_start, rec.t_end, rec.located = t_start, t_end, True
            out.append(rec)
            continue
        setup, no_setup = find_setup(tracks, handler, iv.start_type, t_start, t0, t_end)
        transition = iv.start_type == LIVE and (t_end - t0) < TRANSITION_S
        visible, _ = still_players(tracks, setup)
        out.append(HalfcourtRecord(
            game_id=game_id, index=index, team=team, start_type=iv.start_type, terminal=iv.terminal,
            free_throws=iv.free_throws, clock_start=iv.start_clock, clock_end=iv.end_clock,
            t_start=t_start, t_end=t_end, t0=t0, setup=setup, no_setup=no_setup,
            transition=transition, located=True, outcome=outcome, points=points,
            n_visible_at_setup=visible,
            players=[{"track_id": tr.track_id, "name": tr.name, "jersey": tr.jersey,
                      "trajectory": [[t, round(x, 2), round(y, 2)]
                                     for t, (x, y) in sorted(tr.xy.items())]}
                     for tr in tracks],
            ball_handler=[[t, round(x, 2), round(y, 2)] for t, (x, y) in handler.items()],
            events=[e.to_dict() for e in iv.events],
        ))
    return out
