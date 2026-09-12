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


def test_man_defence_stability_survives_defender_list_order_mismatch():
    # Same man-coverage geometry as test_man_defence_scores_high_stability_and_follow (each
    # defender i shadows attacker i throughout), but the defender dicts are listed in an order
    # that does not match the attacker order or spatial proximity to the rim: defs[0] guards
    # attacker 0 (far from the rim) while defs[1], the *second*-listed defender, guards attacker
    # 2 (much nearer the rim). Both listed defenders have identical trajectory length, so
    # `halfcourt._rank` ties between them and list order alone decides `positions_at`'s output
    # order -- which no longer lines up with anything geometric. `defense.features` must still
    # get stability right here because it identifies matchups by track_id, not by whichever slot
    # positions_at happened to rank a defender into.
    n = 90
    offs = [moving(i, f"A{i}", [(10 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)]
    defs_by_attacker = [
        moving(10 + i, f"D{i}", [(12 + 0.5 * k, 8 * i + 5) for k in range(n)]) for i in range(5)
    ]
    order = [0, 2, 1, 3, 4]  # defs[1] (second listed) guards attacker 2, nearest the rim
    defs = [defs_by_attacker[i] for i in order]
    r = rec(offs, setup=100.0, t_end=110.0)
    r.opponents = defs
    f = D.features(r)
    assert f is not None and f["stability"] > 0.95


def test_label_rule_separates_synthetic_man_and_zone():
    man = [{"stability": 0.95, "follow": 0.9, "spread": 6.0, "gap": 3.0, "paint": 1.5, "frames": 80}] * 10
    zone = [{"stability": 0.4, "follow": 0.1, "spread": 1.0, "gap": 8.0, "paint": 3.5, "frames": 80}] * 10
    labels = D.label_rule(man + zone + [None])
    assert all(l == "man" for l, _ in labels[:10]) and all(l == "zone" for l, _ in labels[10:20])
    assert labels[20] == ("unknown", 0.0)
