"""Period boundaries in video time, from the game clock resetting on the scoreboard."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

# a fresh period's starting clock: 20:00 for a half, 5:00 for overtime
PERIOD_LENGTHS = (1200.0, 300.0)
NEAR_S = 30.0          # a candidate's clock must be within this of a period length
ENDED_BELOW_S = 30.0   # the outgoing period must have counted down to at least this low
CONFIRM_WINDOW_S = 60.0  # look this far ahead to confirm a fresh countdown, not a blip
_K = 5  # width of the trailing/rolling median windows
SCORE_TOL = 12  # a read's score may differ from its neighbours' median by at most this


@dataclass(frozen=True)
class PeriodSpan:
    period: int
    t_lo: float
    t_hi: float


def _score_consistent(trusted_reads, i: int, tol: float = SCORE_TOL) -> bool:
    """Whether read `i`'s scores agree with its neighbours, so it may count toward a run-down.

    `trusted_reads` is a sequence of `(t, clock, away, home)` tuples. For each side (away, home),
    the reference is the median of that side's non-None values among up to `_K` reads before `i`
    and `_K` reads after (excluding `i` itself). A side with no such neighbour has no reference
    and is treated as consistent, which keeps reads with `away=None, home=None` (as in tests that
    don't model scores) trusted. Otherwise the read is inconsistent for that side when its value
    is None or differs from the reference by more than `tol`.
    """
    _, _, away, home = trusted_reads[i]
    lo, hi = max(0, i - _K), min(len(trusted_reads), i + _K + 1)
    neighbours = trusted_reads[lo:i] + trusted_reads[i + 1:hi]

    def side_ok(value, idx):
        refs = [r[idx] for r in neighbours if r[idx] is not None]
        if not refs:
            return True
        return value is not None and abs(value - statistics.median(refs)) <= tol

    return side_ok(away, 2) and side_ok(home, 3)


def period_spans(reads, min_jump_s: float = 200.0, min_span_s: float = 300.0) -> list[PeriodSpan]:
    """Split the scoreboard timeline into periods where the clock resets to a fresh period.

    Real scoreboard OCR is noisy in ways a naive "the clock went up" rule cannot survive: a digit
    misread (`9:05` read as `19:05`), a dropped colon (`4:52` read as `45.2`), and a replay that
    legitimately shows an earlier point in the same period all make a single trusted read look
    like a rise. Comparing against the raw previous read would treat any of these as a period
    boundary; the real first-half timeline splits into 11 spurious periods under that rule. Every
    check below exists to rule out one of those false positives:

    - the rise is measured against `pmed`, the median clock of the previous `_K` trusted reads
      (not the single previous read), so one misread among several good neighbours cannot move
      the baseline enough to look like a jump;
    - the new clock must land within `NEAR_S` of an actual period length (`PERIOD_LENGTHS`): a
      replay jumping back to some arbitrary earlier value in the same period does not qualify,
      only a jump to (about) 20:00 or 5:00 does;
    - the outgoing period must have visibly run down first (a rolling median of `_K` consecutive
      reads since the last accepted start reached `ENDED_BELOW_S` or below) — a period cannot end
      before it's had time to end;
    - a read only counts toward that run-down when its scores are consistent with its neighbours
      (`_score_consistent`): a broadcast graphic that briefly replaces the scoreboard reads
      random digits in the clock box alongside garbage scores, and a handful of such reads can
      otherwise look like a run-down all on their own;
    - at least `min_span_s` of video must have elapsed since the last accepted start, so a run of
      misreads packed close together cannot fabricate a period;
    - the reads in the following `CONFIRM_WINDOW_S` seconds must mostly look like a fresh
      countdown from the new value (each within roughly 15 s low / 5 s high of where a clock
      counting down from it would be); with no reads in that window there is nothing to
      contradict the candidate, so it is accepted.

    The last span ends `5.0` s after the last trusted read. Periods are numbered from 1, and
    spans are contiguous: each span's `t_lo` is the previous span's `t_hi`.
    """
    trusted = sorted(((r.t, r.clock, r.away, r.home) for r in reads if r.clock is not None),
                     key=lambda p: p[0])
    if not trusted:
        return []

    starts = [trusted[0][0]]
    # clocks seen since the last accepted start (inclusive), skipping reads whose scores are
    # inconsistent with their neighbours (see _score_consistent) so a stray graphic's garbage
    # scores cannot fake a run-down
    since_clocks = [trusted[0][1]] if _score_consistent(trusted, 0) else []
    ran_down = len(since_clocks) >= _K and statistics.median(since_clocks[-_K:]) <= ENDED_BELOW_S

    for i in range(1, len(trusted)):
        t, clock, _, _ = trusted[i]
        window = [c for _, c, _, _ in trusted[max(0, i - _K):i]]
        pmed = statistics.median(window)
        accepted = False
        if clock - pmed > min_jump_s:
            near_period_length = any(abs(clock - length) <= NEAR_S for length in PERIOD_LENGTHS)
            enough_span = (t - starts[-1]) >= min_span_s
            if near_period_length and ran_down and enough_span:
                confirm = []
                for tt, cc, _, _ in trusted[i + 1:]:
                    if tt <= t:
                        continue
                    if tt > t + CONFIRM_WINDOW_S:
                        break
                    confirm.append((tt, cc))
                if not confirm:
                    accepted = True
                else:
                    hits = sum(1 for tt, cc in confirm
                              if clock - (tt - t) - 15 <= cc <= clock + 5)
                    accepted = hits / len(confirm) >= 0.7
        if accepted:
            starts.append(t)
            since_clocks = [clock] if _score_consistent(trusted, i) else []
            ran_down = False
        else:
            if _score_consistent(trusted, i):
                since_clocks.append(clock)
            if len(since_clocks) >= _K and statistics.median(since_clocks[-_K:]) <= ENDED_BELOW_S:
                ran_down = True

    end = trusted[-1][0] + 5.0
    bounds = list(zip(starts, starts[1:] + [end]))
    return [PeriodSpan(period=i + 1, t_lo=float(lo), t_hi=float(hi))
            for i, (lo, hi) in enumerate(bounds)]
