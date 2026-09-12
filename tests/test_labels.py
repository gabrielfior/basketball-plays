import json

from basketball_plays import labels as L


def test_load_missing_is_empty_and_save_roundtrips(tmp_path):
    p = tmp_path / "labels.json"
    lab = L.load(p)
    assert lab.clusters == {} and lab.validation == {} and lab.defense == {}
    lab.clusters["3"] = "Horns"; lab.clusters["5"] = "merge:3"; lab.clusters["7"] = "discard"
    L.save(lab, p)
    back = L.load(p)
    assert back.clusters == lab.clusters and json.loads(p.read_text())["version"] == 1


def test_resolve_follows_merges_and_hides_discards():
    lab = L.Labels(clusters={"3": "Horns", "5": "merge:3", "7": "discard", "9": "merge:5"})
    assert L.resolve(lab, "3") == "Horns" and L.resolve(lab, "5") == "Horns"
    assert L.resolve(lab, "9") == "Horns" and L.resolve(lab, "7") is None and L.resolve(lab, "1") is None


def test_apply_maps_records_to_set_names():
    lab = L.Labels(clusters={"0": "Stack"})
    out = L.apply(lab, {"g:1:4": 0, "g:1:5": 2})
    assert out == {"g:1:4": "Stack", "g:1:5": None}
