from basketball_plays import features as F
from basketball_plays.halfcourt import HalfcourtRecord


def rec(players, setup=100.0, t_end=112.0, start_type="dead", full_court=False, handler=None,
        no_setup=False, clock=900.0):
    return HalfcourtRecord(
        game_id="g", index=1, team="Duke", start_type=start_type, full_court=full_court,
        terminal="shot", free_throws=False, clock_start=clock, clock_end=clock - 15,
        t_start=99.0, t_end=t_end, t0=99.5, setup=setup, no_setup=no_setup, transition=False,
        located=True, outcome="made_2", points=2, n_visible_at_setup=len(players),
        players=players, ball_handler=handler or [], events=[],
    )


def player(tid, name, xy_by_t, detected=True):
    rows = [[t, x, y] for t, (x, y) in sorted(xy_by_t.items())]
    return {"track_id": tid, "name": name, "jersey": None, "trajectory": rows,
            "detected": [r[0] for r in rows] if detected else [], "length": len(rows)}


def still(tid, name, x, y, t0=99.0, n=130):
    return player(tid, name, {round(t0 + k / 10, 3): (x, y) for k in range(n)})


def test_bucket_rules():
    assert F.bucket(rec([], start_type="ato")) == "ato"
    assert F.bucket(rec([], start_type="dead", full_court=False)) == "inbound"
    assert F.bucket(rec([], start_type="dead", full_court=True)) == "after_score"


def test_occupancy_by_bin_counts_players_per_zone():
    # five players standing still: top (28,25), both corners, and two blocks
    ps = [still(1, "A", 28.0, 25.0), still(2, "B", 2.0, 2.0), still(3, "C", 2.0, 48.0),
          still(4, "D", 8.0, 17.0), still(5, "E", 8.0, 33.0)]
    occ = F.occupancy_by_bin(rec(ps))
    assert occ.shape == (5, 22)
    from basketball_plays import zones as Z
    for b in range(4):  # bins with frames
        assert abs(occ[b, Z.ZONE_NAMES.index("top")] - 1.0) < 1e-6
        assert abs(occ[b].sum() - 5.0) < 1e-6
    assert abs(occ[4].sum() - 5.0) < 1e-6


def test_handler_zones_modal_per_bin_and_minus_one_when_absent():
    handler = [[100.0 + k / 10, 28.0, 25.0] for k in range(40)]     # bins 1 and 2 at the top
    h = F.handler_zones(rec([still(1, "A", 28.0, 25.0)], handler=handler))
    from basketball_plays import zones as Z
    assert h[1] == Z.ZONE_NAMES.index("top") and h[2] == Z.ZONE_NAMES.index("top")
    assert h[0] == -1 and h[4] == -1


def test_identity_snapshot_marks_missing_players_not_visible():
    ps = [still(1, "Cameron Boozer", 28.0, 25.0)]
    snap = F.identity_snapshot(rec(ps), ["Cameron Boozer", "Isaiah Evans"])
    assert snap.shape == (2, 23)
    from basketball_plays import zones as Z
    assert snap[0, Z.ZONE_NAMES.index("top")] == 1 and snap[0].sum() == 1
    assert snap[1, 22] == 1 and snap[1].sum() == 1


def test_context_buckets():
    c = F.context(rec([], clock=90.0))
    assert c["clock_bucket"] == "late" and c["margin_bucket"] == "unknown"
    assert F.context(rec([], clock=700.0))["clock_bucket"] == "early"


def test_cluster_vector_shape_and_bucket_onehot():
    v = F.cluster_vector(rec([still(1, "A", 28.0, 25.0)], start_type="ato"))
    assert v.shape == (5 * 22 + 5 + 3,)
    assert v[-3:].tolist() == [1.0, 0.0, 0.0]        # ato, inbound, after_score


def test_build_rows_assigns_split_from_game():
    rows = F.build_rows([rec([still(1, "A", 28.0, 25.0)])], {"g": "test"})
    assert len(rows) == 1 and rows[0].split == "test" and rows[0].bucket == "inbound"
    assert rows[0].vector.shape == (118,)


def test_in_scope_keeps_located_dead_ball_halfcourt_records():
    assert F.in_scope(rec([], start_type="dead"))
    assert F.in_scope(rec([], start_type="ato"))
    r = rec([], start_type="live")
    assert not F.in_scope(r)
    r = rec([], start_type="dead")
    r.transition = True
    assert not F.in_scope(r)
    r = rec([], start_type="dead")
    r.located = False
    assert not F.in_scope(r)
    r = rec([], start_type="dead")
    r.t0 = None
    assert not F.in_scope(r)
