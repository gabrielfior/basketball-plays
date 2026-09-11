"""Team naming for appearance clusters and jersey-number resolution per track."""

from __future__ import annotations

import re
from collections import Counter

from basketball_plays.rosters import DUKE, MICHIGAN, ROSTERS


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    digits = re.sub(r"\D", "", str(text))
    if not digits or len(digits) > 2:
        return None
    return str(int(digits))  # strips leading zeros, keeps "0"


def resolve_jersey(reads: list[str | None], roster: dict[str, str], min_votes: int = 2) -> str | None:
    """Plurality vote over OCR reads restricted to numbers that exist on the roster."""
    counts = Counter(n for n in (normalize_number(r) for r in reads) if n is not None and n in roster)
    if not counts:
        return None
    top = counts.most_common(2)
    if top[0][1] < min_votes:
        return None
    if len(top) > 1 and top[1][1] == top[0][1]:
        return None
    return top[0][0]


def _roster_agreement(reads: list[str | None], roster: dict[str, str]) -> int:
    return sum(1 for r in reads if normalize_number(r) in roster)


def name_clusters(
    cluster_brightness: dict[int, float],
    cluster_reads: dict[int, list[str | None]],
    home_team: str = DUKE,
    away_team: str = MICHIGAN,
    min_roster_margin: int = 3,
) -> dict[int, str]:
    """Decide which appearance cluster is which team.

    Roster agreement of OCR reads decides when it is clear-cut (margin >= min_roster_margin);
    otherwise the brighter cluster is the home team, which wears white in NCAA games.
    """
    reads0 = cluster_reads.get(0, [])
    reads1 = cluster_reads.get(1, [])
    option_a = _roster_agreement(reads0, ROSTERS[home_team]) + _roster_agreement(reads1, ROSTERS[away_team])
    option_b = _roster_agreement(reads0, ROSTERS[away_team]) + _roster_agreement(reads1, ROSTERS[home_team])
    if abs(option_a - option_b) >= min_roster_margin:
        home_is_0 = option_a > option_b
    else:
        home_is_0 = cluster_brightness.get(0, 0.0) >= cluster_brightness.get(1, 0.0)
    return {0: home_team, 1: away_team} if home_is_0 else {0: away_team, 1: home_team}
