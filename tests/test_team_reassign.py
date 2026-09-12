from basketball_plays import extract
from basketball_plays.rosters import DUKE, MICHIGAN

# "12" belongs to Duke only here (Michigan's roster has no overlapping numbers).
EXCLUSIVE_ROSTERS = {
    DUKE: {"12": "Cameron Boozer", "3": "Tre Johnson"},
    MICHIGAN: {"23": "Yaxel Lendeborg"},
}

# "12" is shared by both rosters; "3" belongs to Duke only.
SHARED_ROSTERS = {
    DUKE: {"12": "Cameron Boozer", "3": "Tre Johnson"},
    MICHIGAN: {"12": "Some Wolverine", "23": "Yaxel Lendeborg"},
}


def test_flips_to_team_with_unambiguous_reads():
    # Only Duke has jersey 12 among the reads -> flips even though the cluster said Michigan.
    team = extract.reassign_team_by_jersey(["12", "12"], MICHIGAN, EXCLUSIVE_ROSTERS, DUKE, MICHIGAN)
    assert team == DUKE


def test_shared_number_does_not_flip():
    # 12 is on both rosters -> 2 votes each -> no flip, stays the cluster's team.
    team = extract.reassign_team_by_jersey(["12", "12"], MICHIGAN, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == MICHIGAN


def test_mixed_shared_and_unique_number_stays_split_no_flip():
    # "12" matches both rosters, "3" matches Duke only: jersey_votes' plurality vote for each
    # team tops out at 1 (tied candidates within that roster), so neither side reaches
    # MIN_TEAM_VOTES -> no flip.
    team = extract.reassign_team_by_jersey(["12", "3"], MICHIGAN, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == MICHIGAN


def test_duke_only_number_flips_to_duke():
    team = extract.reassign_team_by_jersey(["3", "3"], MICHIGAN, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == DUKE


def test_single_read_never_flips():
    team = extract.reassign_team_by_jersey(["3"], MICHIGAN, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == MICHIGAN


def test_empty_reads_never_flips():
    team = extract.reassign_team_by_jersey([], MICHIGAN, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == MICHIGAN


def test_none_cluster_team_can_still_flip_to_unambiguous_team():
    team = extract.reassign_team_by_jersey(["3", "3"], None, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team == DUKE


def test_none_cluster_team_stays_none_when_ambiguous():
    team = extract.reassign_team_by_jersey(["12", "12"], None, SHARED_ROSTERS, DUKE, MICHIGAN)
    assert team is None


def test_two_different_duke_only_numbers_each_once_does_not_flip():
    # "3" and "12" are both Duke-only (Michigan has neither), but each is read only once. A
    # team's vote is the count of its single most-repeated on-roster number (identity.jersey_votes'
    # plurality), not the sum of all roster-matching reads -- two different numbers seen once
    # each is weaker evidence than one number seen twice under OCR noise, so this stays below
    # MIN_TEAM_VOTES and does not flip.
    team = extract.reassign_team_by_jersey(["3", "12"], MICHIGAN, EXCLUSIVE_ROSTERS, DUKE, MICHIGAN)
    assert team == MICHIGAN


def test_same_duke_only_number_repeated_does_flip():
    # Contrast with the case above: the same number repeated clears MIN_TEAM_VOTES and flips.
    team = extract.reassign_team_by_jersey(["3", "3"], MICHIGAN, EXCLUSIVE_ROSTERS, DUKE, MICHIGAN)
    assert team == DUKE
