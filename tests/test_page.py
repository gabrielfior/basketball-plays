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
