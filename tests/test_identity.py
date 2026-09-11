from basketball_plays import identity
from basketball_plays.rosters import DUKE, MICHIGAN, ROSTERS


def test_normalize_number():
    assert identity.normalize_number(" 12 ") == "12"
    assert identity.normalize_number("07") == "7"
    assert identity.normalize_number("00") == "0"
    assert identity.normalize_number("abc") is None
    assert identity.normalize_number("123") is None
    assert identity.normalize_number(None) is None


def test_resolve_jersey_plurality_on_roster():
    reads = ["12", "12", "17", "1Z", "12", "21"]
    assert identity.resolve_jersey(reads, ROSTERS[DUKE]) == "12"


def test_resolve_jersey_needs_min_votes_and_no_tie():
    assert identity.resolve_jersey(["12"], ROSTERS[DUKE]) is None
    assert identity.resolve_jersey(["12", "12", "21", "21"], ROSTERS[DUKE]) is None
    assert identity.resolve_jersey(["99", "99", "99"], ROSTERS[DUKE]) is None  # not on roster


def test_name_clusters_by_brightness_when_reads_are_ambiguous():
    m = identity.name_clusters({0: 90.0, 1: 200.0}, {0: [], 1: []})
    assert m == {1: DUKE, 0: MICHIGAN}


def test_name_clusters_roster_evidence_overrides_brightness():
    # cluster 0 reads Michigan-only numbers (23, 42, 4) despite being brighter
    reads0 = ["23", "23", "42", "42", "4", "23"]
    reads1 = ["12", "14", "6", "20"]  # Duke-only numbers
    m = identity.name_clusters({0: 220.0, 1: 90.0}, {0: reads0, 1: reads1})
    assert m == {0: MICHIGAN, 1: DUKE}
