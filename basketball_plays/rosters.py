"""Rosters for ESPN game 401817238 (Michigan at Duke, 2026-02-21), jersey -> display name."""

DUKE = "Duke"
MICHIGAN = "Michigan"

TEAM_COLORS_HEX = {DUKE: "#00539B", MICHIGAN: "#FFCB05"}

ROSTERS: dict[str, dict[str, str]] = {
    MICHIGAN: {
        "23": "Yaxel Lendeborg",
        "21": "Morez Johnson Jr.",
        "15": "Aday Mara",
        "3": "Elliot Cadeau",
        "4": "Nimari Burnett",
        "42": "Will Tschetter",
        "2": "L.J. Cason",
        "11": "Roddy Gayle Jr.",
        "1": "Trey McKenney",
        "5": "Oscar Goodman",
        "13": "Harrison Hochberg",
        "32": "Malick Kordel",
        "0": "Ricky Liburd",
        "7": "Howard Eisley Jr.",
        "10": "Winters Grady",
        "12": "Charlie May",
    },
    DUKE: {
        "12": "Cameron Boozer",
        "21": "Patrick Ngongba II",
        "7": "Dame Sarr",
        "3": "Isaiah Evans",
        "1": "Caleb Foster",
        "6": "Maliq Brown",
        "14": "Nikolas Khamenia",
        "2": "Cayden Boozer",
        "15": "Ifeanyi Ufochukwu",
        "5": "Sebastian Wilkins",
        "20": "Jack Scott",
        "8": "Darren Harris",
        "13": "Cameron Sheffield",
    },
}


def player_name(team: str, jersey: str | None) -> str | None:
    if jersey is None:
        return None
    return ROSTERS.get(team, {}).get(jersey)
