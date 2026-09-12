import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "coverage_report", Path(__file__).resolve().parents[1] / "scripts" / "coverage_report.py")
CR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CR)


def cov(periods, expected=2, agree=69, disagree=6, clock_trust=0.87):
    return {"game": "g", "layout": "espn", "expected_periods": expected,
            "detected_periods": len(periods), "period_check": "ok" if len(periods) == expected else "mismatch",
            "espn_agree": agree, "espn_disagree": disagree, "clock_trust_rate": clock_trust,
            "score_trust_rate": clock_trust, "cost_estimate": 6.99,
            "periods": [{"period": i + 1, "intervals": n, "located": loc, "dead_ball": 10,
                         "dead_ball_with_setup": 5, "start_types": {}}
                        for i, (n, loc) in enumerate(periods)]}


def test_acceptance_passes_a_clean_game():
    a = CR.acceptance(cov([(36, 35), (32, 30)]))
    assert a["ok"] and a["periods_ok"] and a["located_ok"] and a["agreement_ok"] and a["clock_ok"]
    assert abs(a["located_rate"] - 65 / 68) < 1e-9


def test_acceptance_flags_each_failure_independently():
    assert not CR.acceptance(cov([(36, 35)], expected=2))["periods_ok"]
    assert not CR.acceptance(cov([(36, 20), (32, 30)]))["located_ok"]
    assert not CR.acceptance(cov([(36, 35), (32, 30)], agree=5, disagree=5))["agreement_ok"]
    assert not CR.acceptance(cov([(36, 35), (32, 30)], clock_trust=0.5))["clock_ok"]


def test_overtime_is_accepted_when_espn_has_three_periods():
    assert CR.acceptance(cov([(36, 35), (32, 30), (8, 8)], expected=3))["periods_ok"]


def test_render_table_marks_pending_and_failing_games():
    rows = [{"game": "1", "opponent": "A", "date": "2026-01-01", "layout": "espn", "status": "ok",
             "periods": 2, "located_rate": 0.95, "agreement_rate": 0.9, "dead_ball": 37, "setups": 19,
             "cost": 5.0, "flags": []},
            {"game": "2", "opponent": "B", "date": "2026-01-02", "layout": "cbs", "status": "pending"}]
    md = CR.render_table(rows)
    assert "| A |" in md and "pending" in md and "95%" in md
