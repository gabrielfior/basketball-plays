"""Teams, ids, rosters and starters of one game, read from its ESPN summary."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameInfo:
    home: str
    away: str
    team_by_id: dict[str, str]
    rosters: dict[str, dict[str, str]]
    starters: dict[str, list[str]]
    periods: list[int]
    colors: dict[str, str] = field(default_factory=dict)


def short_name(display_name: str, mascot: str) -> str:
    """'North Carolina Tar Heels' with mascot 'Tar Heels' -> 'North Carolina'."""
    if mascot and display_name.endswith(mascot):
        return display_name[: -len(mascot)].strip()
    return display_name


def from_summary(summary: dict) -> GameInfo:
    competitors = summary["header"]["competitions"][0]["competitors"]
    names: dict[str, str] = {}
    colors: dict[str, str] = {}
    home = away = None
    for c in competitors:
        team = c["team"]
        name = short_name(team["displayName"], team.get("name", ""))
        names[str(team["id"])] = name
        if team.get("color"):
            colors[name] = "#" + team["color"]
        if c["homeAway"] == "home":
            home = name
        else:
            away = name
    if home is None or away is None:
        raise ValueError("summary has no home/away competitors")
    rosters: dict[str, dict[str, str]] = {n: {} for n in names.values()}
    starters: dict[str, list[str]] = {n: [] for n in names.values()}
    for block in summary.get("boxscore", {}).get("players", []):
        team = names.get(str(block["team"]["id"]))
        if team is None:
            continue
        for stat in block.get("statistics", [])[:1]:
            for a in stat.get("athletes", []):
                ath = a["athlete"]
                jersey = str(ath.get("jersey", "")).strip()
                if jersey:
                    rosters[team][jersey] = ath["displayName"]
                if a.get("starter"):
                    starters[team].append(ath["displayName"])
    periods = sorted({p["period"]["number"] for p in summary.get("plays", []) if "period" in p})
    return GameInfo(home=home, away=away, team_by_id=names, rosters=rosters, starters=starters,
                    periods=periods, colors=colors)
