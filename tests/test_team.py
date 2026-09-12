"""Supervised team classifier and jersey-based crop labelling (basketball_plays/team.py).

Every test here runs on CPU: `TeamClassifier(embedder=...)` injects a fake embedder so torch,
transformers and umap are never imported.
"""

import pickle

import numpy as np
import pytest

from basketball_plays.team import TeamClassifier, label_crops_by_jersey

BLUE = (180, 40, 40)  # BGR, like a dark road uniform
WHITE = (235, 235, 235)


def colour_embedder(crops):
    """Stand-in for SigLIP: a crop's mean BGR, which separates the two uniforms linearly."""
    return np.array([[float(c[..., k].mean()) for k in range(3)] for c in crops], dtype=float)


def a_crop(colour, jitter=0.0, size=16):
    c = np.zeros((size, size, 3), dtype=np.uint8)
    c[:] = np.clip(np.array(colour) + jitter, 0, 255).astype(np.uint8)
    return c


def two_groups(n_blue, n_white):
    """(crops, labels) with `n_blue` home (0) and `n_white` away (1) crops, plus noise crops."""
    crops, labels = [], []
    for i in range(n_blue):
        crops.append(a_crop(BLUE, jitter=i % 5))
        labels.append(0)
    for i in range(n_white):
        crops.append(a_crop(WHITE, jitter=-(i % 5)))
        labels.append(1)
    crops.append(a_crop((120, 120, 120)))  # unlabelled crop: must be ignored by the fit
    labels.append(None)
    return crops, labels


def a_classifier():
    return TeamClassifier(device="cpu", embedder=colour_embedder)


def test_fit_supervised_learns_the_two_uniforms_and_predicts_new_crops():
    clf = a_classifier()
    crops, labels = two_groups(20, 18)
    assert clf.fit_supervised(crops, labels) is True
    assert clf.supervised is True and clf.linear is not None
    pred = clf.predict([a_crop(BLUE, jitter=7), a_crop(WHITE, jitter=-7), a_crop(BLUE)])
    assert list(pred) == [0, 1, 0]
    # brightness is recorded per class, and white is the brighter one
    assert clf.brightness[1] > clf.brightness[0]


def test_fit_supervised_refuses_when_a_class_is_too_small():
    clf = a_classifier()
    crops, labels = two_groups(20, 14)
    assert clf.fit_supervised(crops, labels) is False
    assert clf.linear is None and clf.supervised is False


def test_fit_supervised_refuses_when_a_class_is_absent():
    clf = a_classifier()
    crops, labels = two_groups(30, 0)
    assert clf.fit_supervised(crops, labels) is False
    assert clf.linear is None


def test_fit_supervised_honours_min_per_class():
    clf = a_classifier()
    crops, labels = two_groups(6, 5)
    assert clf.fit_supervised(crops, labels, min_per_class=5) is True
    assert clf.predict([a_crop(WHITE)]).tolist() == [1]


def test_save_load_round_trip_keeps_the_linear_model(tmp_path):
    clf = a_classifier()
    crops, labels = two_groups(20, 20)
    assert clf.fit_supervised(crops, labels) is True
    path = tmp_path / "team_classifier.pkl"
    clf.save(path)

    loaded = TeamClassifier(device="cpu", embedder=colour_embedder).load(path)
    assert loaded.supervised is True and loaded.linear is not None
    assert loaded.brightness == clf.brightness
    assert loaded.predict([a_crop(BLUE), a_crop(WHITE)]).tolist() == [0, 1]


def test_loading_an_old_pickle_without_the_new_keys_falls_back_to_clustering(tmp_path):
    path = tmp_path / "old.pkl"
    with open(path, "wb") as f:
        pickle.dump({"reducer": None, "kmeans": None, "brightness": {0: 1.0, 1: 2.0}}, f)
    loaded = TeamClassifier(device="cpu", embedder=colour_embedder).load(path)
    assert loaded.linear is None and loaded.supervised is False
    assert loaded.brightness == {0: 1.0, 1: 2.0}


def test_predict_on_no_crops_is_empty():
    clf = a_classifier()
    crops, labels = two_groups(20, 20)
    clf.fit_supervised(crops, labels)
    assert clf.predict([]).tolist() == []


def test_the_fake_embedder_means_no_gpu_imports_are_needed():
    # constructing with an embedder must not import torch/transformers/umap
    import sys

    a_classifier()
    assert "torch" not in sys.modules and "umap" not in sys.modules


HOME = ["0", "2", "23", "35"]
AWAY = ["1", "3", "23", "40"]


@pytest.mark.parametrize("read, expected", [
    ("2", 0),        # home only
    ("35", 0),
    ("1", 1),        # away only
    ("40", 1),
    ("23", None),    # on both rosters
    ("99", None),    # on neither
    ("", None),      # no read at all
    (None, None),
    ("garbage", None),
    ("#23", None),   # normalises to 23, still shared
    ("#35", 0),      # punctuation is stripped
    ("35 ", 0),      # whitespace is stripped
    ("02", 0),       # leading zero -> 2
    ("0", 0),        # number zero is a real jersey, not a falsy blank
    ("123", None),   # three digits is never a jersey
])
def test_label_crops_by_jersey(read, expected):
    assert label_crops_by_jersey([read], HOME, AWAY) == [expected]


def test_label_crops_by_jersey_labels_a_whole_batch_in_order():
    reads = ["2", "1", "23", "", "40"]
    assert label_crops_by_jersey(reads, HOME, AWAY) == [0, 1, None, None, 1]


def test_label_crops_by_jersey_normalises_the_roster_numbers_too():
    # ESPN rosters spell some numbers with a leading zero
    assert label_crops_by_jersey(["4"], ["04"], ["5"]) == [0]
    assert label_crops_by_jersey(["04"], ["4"], ["5"]) == [0]
