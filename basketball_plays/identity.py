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


def jersey_votes(reads: list[str | None], roster: dict[str, str], min_votes: int = 2) -> tuple[str | None, int]:
    """Plurality vote over OCR reads restricted to numbers on the roster: (jersey, votes)."""
    counts = Counter(n for n in (normalize_number(r) for r in reads) if n is not None and n in roster)
    if not counts:
        return None, 0
    top = counts.most_common(2)
    if top[0][1] < min_votes:
        return None, top[0][1]
    if len(top) > 1 and top[1][1] == top[0][1]:
        return None, top[0][1]
    return top[0][0], top[0][1]


def resolve_jersey(reads: list[str | None], roster: dict[str, str], min_votes: int = 2) -> str | None:
    return jersey_votes(reads, roster, min_votes)[0]


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


def resolve_conflicts(tracks: list[dict]) -> dict[int, str | None]:
    """Drop duplicate jerseys among concurrent tracks of the same team, keeping the strongest vote.

    Each track dict has: id, team, jersey (may be None), votes, start, end.
    """
    keep = {t["id"]: t["jersey"] for t in tracks}
    ranked = sorted((t for t in tracks if t["jersey"] is not None), key=lambda t: -t["votes"])
    for i, a in enumerate(ranked):
        if keep[a["id"]] is None:
            continue
        for b in ranked[i + 1:]:
            if keep[b["id"]] is None or b["team"] != a["team"] or b["jersey"] != a["jersey"]:
                continue
            if b["start"] <= a["end"] and a["start"] <= b["end"]:
                keep[b["id"]] = None
    return keep
