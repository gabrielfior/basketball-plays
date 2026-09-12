"""Period boundaries in video time, from the game clock resetting on the scoreboard."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodSpan:
    period: int
    t_lo: float
    t_hi: float


def period_spans(reads, min_jump_s: float = 300.0, min_span_s: float = 120.0) -> list[PeriodSpan]:
    """Split the scoreboard timeline into periods where the clock jumps back up.

    Walks trusted clock reads (`clock is not None`) in time order. Any rise over the previous
    trusted read is a candidate new-period start: a rise bigger than `min_jump_s` (a real period
    boundary, e.g. 1 second left in a half followed by a fresh 20:00) is always accepted; a
    smaller rise is only accepted when at least `min_span_s` has passed since the last accepted
    start, so a single misread (the clock briefly reading high, then correcting) cannot open a
    period on its own. The last span ends 5 s after the last trusted read. Period numbers start
    at 1.
    """
    trusted = sorted(((r.t, r.clock) for r in reads if r.clock is not None), key=lambda p: p[0])
    if not trusted:
        return []
    starts = [trusted[0][0]]
    prev_clock = trusted[0][1]
    for t, clock in trusted[1:]:
        if clock > prev_clock:
            forced = clock - prev_clock > min_jump_s
            if forced or t - starts[-1] >= min_span_s:
                starts.append(t)
        prev_clock = clock
    end = trusted[-1][0] + 5.0
    bounds = list(zip(starts, starts[1:] + [end]))
    return [PeriodSpan(period=i + 1, t_lo=float(lo), t_hi=float(hi))
            for i, (lo, hi) in enumerate(bounds)]
