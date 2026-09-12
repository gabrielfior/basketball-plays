"""Tests for the per-game ingest driver's pure helpers (scripts/ingest_game.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

from basketball_plays import scoreboard as sb
from basketball_plays.games import Game, GamePaths
from basketball_plays.periods import PeriodSpan
from basketball_plays.schema import Possession

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ingest = load_script("ingest_game")


def a_game():
    return Game(espn_id="401817238", youtube_id="a" * 11, date="2025-01-01", opponent="Michigan",
                home=True, broadcaster="ESPN", layout="espn", split="train")


def a_summary(period_numbers):
    return {
        "header": {"competitions": [{"competitors": [
            {"homeAway": "home",
             "team": {"id": "150", "displayName": "Duke Blue Devils", "name": "Blue Devils"}},
            {"homeAway": "away",
             "team": {"id": "130", "displayName": "Michigan Wolverines", "name": "Wolverines"}},
        ]}]},
        "plays": [{"period": {"number": n}} for n in period_numbers],
    }


def two_period_timeline():
    """A clock counting 20:00 down to 0:00 twice - two periods to `periods.period_spans`."""
    reads = []
    for period in range(2):
        for i in range(1200):
            t = float(period * 1200 + i)
            reads.append(sb.ScoreboardRead(t=t, clock=1200.0 - i, clock_text=None, away=0, home=0))
    return reads


def a_game_dir(tmp_path, period_numbers=(1, 2)):
    paths = GamePaths.for_game("401817238", root=tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.espn_summary.write_text(json.dumps(a_summary(period_numbers)))
    sb.write_timeline(paths.scoreboard_raw, two_period_timeline())
    return paths


def test_force_is_expanded_transitively_to_every_downstream_step():
    # Redoing the OCR invalidates everything computed from the scoreboard timeline.
    assert ingest.expand_force({"ocr"}) == {"ocr", "extract", "annotate", "halfcourt"}
    assert ingest.expand_force({"download"}) == set(ingest.STEPS)
    assert ingest.expand_force({"halfcourt"}) == {"halfcourt"}
    assert ingest.expand_force(set()) == set()


def test_downstream_graph_only_names_known_steps():
    assert set(ingest.DOWNSTREAM) == set(ingest.STEPS)
    for step, later in ingest.DOWNSTREAM.items():
        assert set(later) <= set(ingest.STEPS)
        assert step not in later


def test_check_periods_passes_when_the_detected_count_matches_espn(tmp_path):
    paths = a_game_dir(tmp_path, period_numbers=(1, 2))

    _, info, spans, fields = ingest.check_periods(a_game(), paths, allow_mismatch=False)

    assert len(spans) == 2
    assert [s.period for s in spans] == [1, 2]
    assert info.home == "Duke"
    assert fields == {"expected_periods": 2, "detected_periods": 2, "period_check": "ok"}


def test_check_periods_exits_2_and_names_the_counts_and_boundaries(tmp_path, capsys):
    # ESPN says the game went to overtime; the scoreboard timeline only yields two periods.
    paths = a_game_dir(tmp_path, period_numbers=(1, 2, 3))

    with pytest.raises(SystemExit) as excinfo:
        ingest.check_periods(a_game(), paths, allow_mismatch=False)

    assert excinfo.value.code == 2
    out = capsys.readouterr().out
    assert "401817238" in out
    assert "2 period(s)" in out and "has 3" in out
    assert "0.0-1200.0s" in out and "1200.0-2404.0s" in out


def test_check_periods_continues_and_records_the_mismatch_when_allowed(tmp_path):
    paths = a_game_dir(tmp_path, period_numbers=(1, 2, 3))

    _, _, spans, fields = ingest.check_periods(a_game(), paths, allow_mismatch=True)

    assert len(spans) == 2
    assert fields == {"expected_periods": 3, "detected_periods": 2, "period_check": "mismatch"}


def a_read(t, clock=None, away=None, home=None):
    return sb.ScoreboardRead(t=float(t), clock=clock, clock_text=None, away=away, home=home)


def test_trust_rates_are_the_fractions_of_reads_that_survived_cleaning():
    good = [a_read(i, clock=1200.0 - i, away=10, home=12) for i in range(6)]
    no_clock = [a_read(i, clock=None, away=10, home=12) for i in range(6, 10)]
    nothing = [a_read(i) for i in range(10, 12)]

    assert ingest.trust_rates(good) == (1.0, 1.0)
    assert ingest.trust_rates(good + no_clock) == (0.6, 1.0)  # score still read on all ten
    assert ingest.trust_rates(good + no_clock + nothing) == (0.5, 10 / 12)
    assert ingest.trust_rates([a_read(0, clock=5.0, away=10, home=None)]) == (1.0, 0.0)
    assert ingest.trust_rates([]) == (0.0, 0.0)


def test_trust_rates_are_computed_per_period_by_cleaning_each_buffered_span_separately():
    # step_halfcourt never cleans the whole game in one pass: for each period it slices the raw
    # timeline down to that period's own (t_lo, t_hi) span, with a +-5s buffer for clock drift at
    # the edges, and cleans only that slice. half2 is a genuine half-time reset (the clock
    # restarts at 1200s), so a naive whole-game clean would wrongly reject nearly all of it.
    half1 = [a_read(i, clock=1200.0 - i, away=0, home=0) for i in range(600)]
    half2 = [a_read(600 + i, clock=1200.0 - i, away=0, home=0) for i in range(600)]
    reads_raw = half1 + half2
    spans = [(0, 599), (600, 1199)]

    trust = ingest.per_period_trust(reads_raw, spans)

    # (a): the helper's numbers are exactly what cleaning each buffered slice independently
    # gives -- built here from the same primitives (`sb.clean_timeline`, `ingest.trust_rates`)
    # but not by calling `per_period_trust` itself, so a bug that cleaned the concatenated
    # timeline once and sliced afterwards (rather than slicing first) would be caught.
    expected = [
        ingest.trust_rates(sb.clean_timeline([r for r in reads_raw if lo - 5 <= r.t <= hi + 5]))
        for lo, hi in spans
    ]
    assert trust == expected

    # (b): a genuine reset never pollutes either period's own trust when cleaned per span --
    # the property a whole-game pass could not guarantee.
    assert all(clock_rate > 0.95 for clock_rate, _ in trust)


def test_game_trust_rates_weight_each_period_by_its_read_count():
    # 900 reads at 1.0 and 100 at 0.0 average to 0.9, not to 0.5.
    assert ingest.weighted_trust_rates([(900, 1.0, 0.8), (100, 0.0, 0.3)]) == {
        "clock_trust_rate": 0.9, "score_trust_rate": 0.75}
    assert ingest.weighted_trust_rates([(0, 0.0, 0.0)]) == {"clock_trust_rate": 0.0,
                                                            "score_trust_rate": 0.0}
    assert ingest.weighted_trust_rates([]) == {"clock_trust_rate": 0.0, "score_trust_rate": 0.0}


def a_possession(pid, espn_points, scoreboard_points):
    return Possession(possession_id=pid, start_time=0.0, end_time=1.0, fps=10.0,
                      offense_team="Duke", attacking_basket="left", players=[], ball=[],
                      points_scored=espn_points, scoreboard_points=scoreboard_points)


def test_espn_agreement_counts_only_possessions_with_both_numbers():
    possessions = [a_possession(0, 2, 2), a_possession(1, 3, 2), a_possession(2, 2, None),
                   a_possession(3, None, 2), a_possession(4, 0, 0)]

    assert ingest.espn_agreement(possessions) == {"espn_agree": 2, "espn_disagree": 1}
    assert ingest.espn_agreement([]) == {"espn_agree": 0, "espn_disagree": 0}


def test_video_duration_rejects_an_unreadable_ffprobe_duration(tmp_path, monkeypatch):
    class Probe:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: Probe("N/A\n"))
    with pytest.raises(SystemExit) as excinfo:
        ingest.video_duration(tmp_path / "video.mp4")
    assert "'N/A'" in str(excinfo.value)

    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: Probe("\n"))
    with pytest.raises(SystemExit):
        ingest.video_duration(tmp_path / "video.mp4")

    monkeypatch.setattr(ingest.subprocess, "run", lambda *a, **k: Probe("4642.5\n"))
    assert ingest.video_duration(tmp_path / "video.mp4") == 4642.5


def test_halfcourt_warns_when_the_spans_changed_since_extract_ran(tmp_path, capsys):
    paths = GamePaths.for_game("401817238", root=tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    used = [PeriodSpan(1, 0.0, 1200.0), PeriodSpan(2, 1200.0, 2400.0)]
    (paths.root / "spans_used.json").write_text(json.dumps(ingest.spans_as_json(used)))

    ingest.warn_if_spans_changed(paths, used)
    assert capsys.readouterr().out == ""

    ingest.warn_if_spans_changed(paths, [PeriodSpan(1, 0.0, 2400.0)])
    out = capsys.readouterr().out
    assert "period spans changed" in out and "--force extract" in out


def test_no_warning_when_extract_never_recorded_its_spans(tmp_path, capsys):
    paths = GamePaths.for_game("401817238", root=tmp_path)
    paths.root.mkdir(parents=True, exist_ok=True)

    ingest.warn_if_spans_changed(paths, [PeriodSpan(1, 0.0, 1200.0)])

    assert capsys.readouterr().out == ""


def test_cost_line_is_parsed_from_the_gpu_step_output():
    line = "estimated cost: 77.7 video minutes x $0.09 = $6.99\n"
    assert ingest.COST_LINE.search(line).group(1) == "6.99"
    assert ingest.COST_LINE.search("nothing here") is None
