from basketball_plays import scoreboard as sb


def test_parse_clock_formats():
    assert sb.parse_clock("19:47") == (19 * 60 + 47, "19:47")
    assert sb.parse_clock("8:49") == (8 * 60 + 49, "8:49")
    assert sb.parse_clock("08") == (0.8, "0.8")  # tesseract drops the decimal point
    assert sb.clock_candidates("311") == [(191, "3:11"), (31.1, "31.1")]  # colon or point dropped
    assert sb.clock_candidates("1947") == [(19 * 60 + 47, "19:47")]
    assert sb.parse_clock("45.3") == (45.3, "45.3")
    assert sb.parse_clock("99:99") == (None, None)
    assert sb.parse_clock("") == (None, None)


def test_parse_score():
    assert sb.parse_score("26") == 26
    assert sb.parse_score("2 6") == 26
    assert sb.parse_score("") is None
    assert sb.parse_score("1234") is None


def reads(rows):
    return [sb.ScoreboardRead(t=float(i), clock=c, clock_text=None, away=a, home=h) for i, (c, a, h) in enumerate(rows)]


def test_clean_timeline_rejects_glitches_and_decreases():
    rows = [
        (600, 10, 8), (599, 10, 8), (598, 70, 8), (597, 12, 8), (596, 12, 8),
        (300, 12, 8), (595, 12, 3), (594, 12, 8),
    ]
    out = sb.clean_timeline(reads(rows))
    assert [r.away for r in out] == [10, 10, None, 12, 12, 12, 12, 12]
    assert [r.home for r in out] == [8, 8, 8, 8, 8, 8, None, 8]
    assert [r.clock for r in out] == [600, 599, 598, 597, 596, None, 595, 594]


def test_clean_timeline_allows_stopped_clock_and_none():
    rows = [(500, 5, 5), (500, 5, 5), (500, None, 5), (500, 5, None), (499, 5, 5)]
    out = sb.clean_timeline(reads(rows))
    assert [r.clock for r in out] == [500, 500, 500, 500, 499]
    assert [r.away for r in out] == [5, 5, None, 5, 5]


def test_value_at_picks_nearest_trusted_read():
    rs = reads([(600, 10, 8), (None, None, None), (598, 12, 8)])
    assert sb.value_at(rs, 1.0, "clock") == 600
    assert sb.value_at(rs, 1.6, "away") == 12
    assert sb.value_at(rs, 30.0, "away") is None


def test_clean_timeline_with_play_by_play_states_rejects_systematic_misreads():
    states = [(0, 0), (2, 0), (4, 0), (4, 2)]
    rows = [(600, 0, 0), (599, 0, 0), (598, 2, 0), (597, 2, 0), (596, 7, 0), (595, 7, 0), (594, 4, 0),
            (593, 4, 0), (592, 4, 2), (591, 4, 2), (590, 2, 0)]
    out = sb.clean_timeline(reads(rows), valid_states=states)
    assert [(r.away, r.home) for r in out] == [
        (0, 0), (0, 0), (2, 0), (2, 0), (None, None), (None, None), (4, 0), (4, 0), (4, 2), (4, 2), (None, None)
    ]


def test_clean_timeline_relocks_after_a_stray_low_block():
    # 19:44 counting down, then a 9-read stray "3.0" graphic, then 19:30 counting down for real.
    clocks = [19 * 60 + 44 - i for i in range(10)]
    clocks += [3.0] * 9
    clocks += [19 * 60 + 30 - i for i in range(20)]
    rows = [(c, 0, 0) for c in clocks]
    out = sb.clean_timeline(reads(rows))
    stray = out[10:19]
    real = out[19:39]
    assert all(r.clock is None for r in stray)  # drop is implausible and never confirmed
    assert all(r.clock is not None for r in real)  # accepted: a small, neighbour-confirmed drop


def test_clean_timeline_accepts_a_real_large_drop_after_a_replay():
    # 19:44 counting down, a 30 s gap with no OCR reads, then 14:10 counting down for real.
    clocks = (
        [19 * 60 + 44 - i for i in range(5)] + [None] * 30 + [14 * 60 + 10 - i for i in range(10)]
    )
    rows = [(c, 0, 0) for c in clocks]
    out = sb.clean_timeline(reads(rows))
    replay = out[35:45]
    assert all(r.clock is not None for r in replay)  # confirmed by the following reads
    assert [r.clock for r in replay] == clocks[35:45]


def test_clean_timeline_still_rejects_an_isolated_spike():
    clocks = [9 * 60 + 9 - i for i in range(4)] + [19 * 60 + 5] + [9 * 60 + 4 - i for i in range(5)]
    rows = [(c, 0, 0) for c in clocks]
    out = sb.clean_timeline(reads(rows))
    assert out[4].clock is None


def test_clean_timeline_disambiguates_dropped_colon_by_context():
    rows = [(120, 5, 5), (119, 5, 5), (118, 5, 5)]
    rs = reads(rows)
    rs[1].clock, rs[1].clock_text = 15.9, "15.9"  # "1:59" read as "159" and saved as tenths
    out = sb.clean_timeline(rs)
    assert [r.clock for r in out] == [120, 119, 118]
    # ...and under one minute the tenths reading wins
    rows = [(45.0, 5, 5), (44.1, 5, 5), (43.2, 5, 5)]
    rs = reads(rows)
    for r, txt in zip(rs, ["45.0", "44.1", "43.2"]):
        r.clock_text = txt
    rs[1].clock_text = "441"
    out = sb.clean_timeline(rs)
    assert [r.clock for r in out] == [45.0, 44.1, 43.2]
