import numpy as np

from basketball_plays import defense as D
from tests.test_features import player, rec


def moving(tid, name, path):
    return player(tid, name, {round(100.0 + k / 10, 3): xy for k, xy in enumerate(path)})


def test_matchups_is_a_one_to_one_assignment_by_distance():
    off = np.array([[10, 10], [10, 40], [30, 25], [20, 5], [20, 45]], float)
    deff = off + np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]])
    m = D.matchups(off, deff)
    assert sorted(m) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert D.matchups(off[:3], deff[:3]) == []


def test_man_defence_scores_high_stability_and_follow():
    # attackers walk right 0.5 ft per frame; defenders shadow them 2 ft behind
    n = 90
    offs = [moving(i, f"A{i}", [(10 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)]
    defs = [moving(10 + i, f"D{i}", [(12 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)]
    r = rec(offs, setup=100.0, t_end=110.0)
    r.opponents = defs
    f = D.features(r)
    assert f is not None and f["stability"] > 0.95 and f["follow"] > 0.9 and f["gap"] < 3


def test_zone_defence_scores_low_follow_and_low_spread():
    n = 90
    offs = [moving(i, f"A{i}", [(10 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)]
    defs = [moving(10 + i, f"D{i}", [(8, 8 * i + 6) for _ in range(n)]) for i in range(5)]  # parked
    r = rec(offs, setup=100.0, t_end=110.0)
    r.opponents = defs
    f = D.features(r)
    assert f is not None and f["spread"] < 0.5 and f["follow"] < 0.3


def alternating_detected(tid, name, path):
    """Like `moving`, but `detected` only holds the even-indexed frames' times: the trajectory
    itself is complete (every frame has a real position), only the detection flag flickers."""
    rows = [[round(100.0 + k / 10, 3), x, y] for k, (x, y) in enumerate(path)]
    detected = [r[0] for k, r in enumerate(rows) if k % 2 == 0]
    return {"track_id": tid, "name": name, "jersey": None, "trajectory": rows,
            "detected": detected, "length": len(rows)}


def test_man_defence_stability_survives_positions_at_index_churn():
    # Same man-coverage geometry as test_man_defence_scores_high_stability_and_follow (defender i
    # shadows attacker i throughout, 2 ft ahead in x), but defender D0's `detected` flag flickers
    # every other frame while its trajectory (and every other track's) covers every frame.
    # `halfcourt._rank` ranks "detected at t" before "not detected at t", so on frames where D0 is
    # marked undetected, positions_at's tie-break demotes it behind every other (always-detected)
    # defender -- the array index `positions_at` assigns to D0 alternates between frames even
    # though D0 is, physically, the same track guarding the same attacker the whole time. An
    # index-based matchup (treating `matchups`'s array positions as identity) would see the
    # (attacker, defender)-index pair for attacker 0 flip every single frame and read that as
    # total instability; `defense.features` must still score this as stable man coverage because
    # it identifies matchups by track_id, not by whichever slot positions_at ranked a defender
    # into on a given frame.
    n = 90
    offs = [moving(i, f"A{i}", [(10 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)]
    d0 = alternating_detected(10, "D0", [(12 + 0.5 * k, 5) for k in range(n)])
    rest = [moving(10 + i, f"D{i}", [(12 + 0.5 * k, 8 * i + 5) for k in range(n)])
            for i in range(1, 5)]
    r = rec(offs, setup=100.0, t_end=110.0)
    r.opponents = [d0] + rest
    f = D.features(r)
    assert f is not None and f["stability"] > 0.95


def test_label_rule_separates_synthetic_man_and_zone():
    man = [{"stability": 0.95, "follow": 0.9, "spread": 6.0, "gap": 3.0, "paint": 1.5, "frames": 80}] * 10
    zone = [{"stability": 0.4, "follow": 0.1, "spread": 1.0, "gap": 8.0, "paint": 3.5, "frames": 80}] * 10
    labels = D.label_rule(man + zone + [None])
    assert all(l == "man" for l, _ in labels[:10]) and all(l == "zone" for l, _ in labels[10:20])
    assert labels[20] == ("unknown", 0.0)
