import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_page", Path(__file__).resolve().parents[1] / "scripts" / "build_page.py")
BP = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BP)


def test_render_page_inlines_data_and_has_all_tabs():
    html = BP.render_page(
        clusters={"k": 2, "clusters": [{"cluster": 0, "n": 3, "n_by_bucket": {}, "n_by_split": {},
                                          "top_zones": ["top"], "handler_path": ["top"] * 5,
                                          "members": [("g", 1, 4, 0.1)]}], "labels": {"g:1:4": 0}},
        defense={"g:1:4": {"defense": "man", "confidence": 0.8}},
        animations={"g:1:4": {"fps": 5, "duke": [[[28.0, 25.0]]], "opp": [[[30.0, 25.0]]], "handler": [[28.0, 25.0]]}},
        meta={"g:1:4": {"bucket": "inbound", "points": 2, "split": "train", "game": "g", "opponent": "X"}},
        montages={0: "data:image/png;base64,AAAA"}, labels={"clusters": {}, "validation": {}, "defense": {}, "version": 1},
        validation_keys=["g:1:4"],
    )
    for tab in ("Clusters", "Validation", "Results", "Browse"):
        assert f">{tab}<" in html
    assert "data:image/png;base64,AAAA" in html and '"g:1:4"' in html
    assert "localStorage" in html and "<script src=" not in html   # offline: no external scripts


def test_sample_animation_carries_matchup_pairs_per_frame():
    from tests.test_features import rec, still

    # five Duke players and five opponents, each opponent parked 2 ft to the right of an attacker
    duke = [still(i, f"A{i}", 10.0, 8.0 * i + 5) for i in range(5)]
    opp = [still(10 + i, f"D{i}", 12.0, 8.0 * i + 5) for i in range(5)]
    r = rec(duke, setup=100.0, t_end=112.0)
    r.opponents = opp
    a = BP.sample_animation(r, fps=5.0, pre=1.0, post=2.0)

    assert len(a["pairs"]) == len(a["duke"])
    tracked = [i for i, f in enumerate(a["duke"]) if len(f) == 5]
    assert tracked
    for i in tracked:
        pairs = a["pairs"][i]
        assert len(pairs) == 5
        assert sorted(p[0] for p in pairs) == [0, 1, 2, 3, 4]      # one-to-one, into this frame
        assert sorted(p[1] for p in pairs) == [0, 1, 2, 3, 4]
        for oi, di in pairs:                                       # nearest defender, 2 ft away
            o, d = a["duke"][i][oi], a["opp"][i][di]
            assert abs(o[1] - d[1]) < 0.01 and abs(d[0] - o[0] - 2.0) < 0.01


def test_sample_animation_has_no_pairs_when_a_side_is_too_thin():
    from tests.test_features import rec, still

    r = rec([still(1, "A", 10.0, 25.0)], setup=100.0, t_end=112.0)
    r.opponents = [still(10 + i, f"D{i}", 12.0, 8.0 * i + 5) for i in range(5)]
    a = BP.sample_animation(r, fps=5.0, pre=1.0, post=2.0)
    assert all(p == [] for p in a["pairs"])
