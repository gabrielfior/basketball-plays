from basketball_plays import periods as P
from basketball_plays.scoreboard import ScoreboardRead


def reads(pairs):
    return [ScoreboardRead(t=float(t), clock=c, clock_text=None, away=None, home=None)
            for t, c in pairs]


def test_two_halves_are_split_where_the_clock_jumps_back_up():
    rs = reads([(0, 1200), (600, 600), (1199, 1)] + [(1300, None)] +
               [(1400, 1200), (2000, 600), (2599, 1)])
    spans = P.period_spans(rs)
    assert [(s.period, s.t_lo, s.t_hi) for s in spans] == [(1, 0.0, 1400.0), (2, 1400.0, 2604.0)]


def test_a_single_misread_does_not_start_a_period():
    rs = reads([(0, 1200), (100, 1100), (101, 1190), (102, 1098), (1000, 200)])
    spans = P.period_spans(rs)
    assert len(spans) == 1 and spans[0].period == 1


def test_overtime_is_a_third_span():
    rs = reads([(0, 1200), (1199, 1), (1300, 1200), (2499, 1), (2600, 300), (2899, 1)])
    assert [s.period for s in P.period_spans(rs)] == [1, 2, 3]
    assert P.period_spans(rs)[2].t_lo == 2600.0


def test_untrusted_reads_are_ignored():
    rs = reads([(0, 1200), (1, None), (2, None), (3, 1197)])
    assert len(P.period_spans(rs)) == 1
