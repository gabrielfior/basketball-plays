from basketball_plays import periods as P
from basketball_plays.scoreboard import ScoreboardRead


def reads(pairs):
    return [ScoreboardRead(t=float(t), clock=c, clock_text=None, away=None, home=None)
            for t, c in pairs]


def countdown(t0, clock0, n):
    """[(t, clock)] counting down one second at a time, one trusted read per second."""
    return [(t0 + k, clock0 - k) for k in range(n)]


def scored(pairs):
    return [ScoreboardRead(t=float(t), clock=c, clock_text=None, away=a, home=h)
            for t, c, a, h in pairs]


def scored_countdown(t0, clock0, n, away, home):
    """[(t, clock, away, home)] counting down one second at a time, constant scores."""
    return [(t0 + k, clock0 - k, away, home) for k in range(n)]


def test_two_halves_are_split_where_the_clock_jumps_back_up():
    rs = reads(countdown(0, 1200, 60) + countdown(1140, 60, 60) +
               countdown(1400, 1200, 60) + countdown(2540, 60, 60))
    spans = P.period_spans(rs)
    assert [(s.period, s.t_lo, s.t_hi) for s in spans] == [(1, 0.0, 1400.0), (2, 1400.0, 2604.0)]


def test_a_digit_misread_does_not_start_a_period():
    rs = reads(countdown(0, 1200, 100) + [(100, 1190)] + countdown(101, 1099, 60))
    spans = P.period_spans(rs)
    assert len(spans) == 1 and spans[0].period == 1


def test_a_replay_showing_an_earlier_clock_does_not_start_a_period():
    rs = reads(countdown(0, 1200, 600) + countdown(600, 900, 20) + countdown(620, 599, 300))
    spans = P.period_spans(rs)
    assert len(spans) == 1 and spans[0].period == 1


def test_a_dropped_colon_read_does_not_start_a_period():
    rs = reads(countdown(0, 1200, 400) + [(400, 45.2)] + countdown(401, 799, 300))
    spans = P.period_spans(rs)
    assert len(spans) == 1 and spans[0].period == 1


def test_overtime_is_a_third_span():
    rs = reads(countdown(0, 1200, 30) + countdown(1170, 30, 30) +
               countdown(1300, 1200, 30) + countdown(2470, 30, 30) +
               countdown(2600, 300, 30) + countdown(2870, 30, 30))
    assert [s.period for s in P.period_spans(rs)] == [1, 2, 3]
    assert P.period_spans(rs)[2].t_lo == 2600.0


def test_untrusted_reads_are_ignored():
    rs = reads([(0, 1200), (1, None), (2, None), (3, 1197)])
    assert len(P.period_spans(rs)) == 1


def test_stray_graphic_reads_with_garbage_scores_do_not_fake_a_run_down():
    rs = scored(scored_countdown(0, 1200, 700, 30, 40) +
               [(701, 2.9, 1, 10), (702, 1.1, 3, None), (703, 1.5, None, 829)] +
               scored_countdown(704, 328, 200, 30, 40))
    spans = P.period_spans(rs)
    assert len(spans) == 1


def test_a_real_period_end_with_consistent_scores_still_counts():
    rs = scored(scored_countdown(0, 1200, 700, 30, 40) +
               [(701, 2.9, 30, 40), (702, 1.1, 30, 40), (703, 1.5, 30, 40)] +
               scored_countdown(704, 1200, 200, 30, 40))
    spans = P.period_spans(rs)
    assert [s.period for s in spans] == [1, 2]
    assert spans[1].t_lo == 704.0
