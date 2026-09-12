"""Scoreboard graphic layouts per broadcaster: where the clock and the two scores sit in a
1280 x 720 frame. Tune with scripts/probe_scoreboard.py; each layout has a fixture frame under
tests/fixtures/scoreboards/."""

from __future__ import annotations

from dataclasses import dataclass, field

from basketball_plays.scoreboard import REGIONS as ESPN_REGIONS

Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class Layout:
    name: str
    regions: dict[str, Region] = field(default_factory=dict)
    invert: bool = False
    note: str = ""
    # Region sets to fall back to (in order) when `regions` doesn't yield a parseable clock,
    # for a broadcast that alternates between two scoreboard graphics. See scoreboard.read_frame.
    alternatives: tuple[dict[str, Region], ...] = ()


LAYOUTS: dict[str, Layout] = {
    "espn": Layout("espn", dict(ESPN_REGIONS), note="ESPN, ESPN2, ACC Network bottom-centre bar"),
    # CBS: one wide bar along the bottom, each team's score in its own colour panel, the clock
    # right of "1ST" on the dark right-hand block.
    "cbs": Layout("cbs", {"away": (495, 610, 560, 658), "clock": (1030, 613, 1128, 658),
                          "home": (892, 612, 958, 658)},
                  note="CBS bottom-left bar (fixture cbs_fsu.jpg)"),
    # NCAA tournament: the broadcast alternates two graphics. Most of the time it's a bottom
    # bar (the primary regions below); during other stretches it's a stacked box at the bottom
    # left, away team on top, home below, and a light strip under both carrying "1ST HALF" then
    # the clock (kept as the alternative). Neither region set reads the other graphic.
    "ncaa": Layout("ncaa", {"away": (418, 636, 470, 676), "clock": (862, 636, 952, 676),
                            "home": (752, 636, 806, 676)},
                   alternatives=({"away": (205, 483, 292, 547), "clock": (200, 635, 276, 658),
                                  "home": (205, 565, 292, 630)},),
                   note="NCAA tournament bottom bar (fixture ncaa_bar_siena_1800.jpg), falling "
                        "back to the stacked box (fixture ncaa_siena.jpg)"),
    # The CW: bottom bar sitting lower in the frame than the others; the home score is on a
    # narrow blue panel, so its box stays inside that panel.
    "cw": Layout("cw", {"away": (548, 650, 615, 696), "clock": (975, 652, 1045, 695),
                        "home": (884, 651, 946, 696)},
                 note="The CW bottom bar (fixture cw_wake.jpg)"),
    # CBS Sports Network: same family as CBS but shifted, with the clock in a light grey panel
    # left of the shot clock. The NHL ticker below the bar stays out of every box.
    "cbssn": Layout("cbssn", {"away": (492, 613, 556, 661), "clock": (1038, 618, 1116, 659),
                              "home": (890, 613, 962, 661)},
                    note="CBS Sports Network bottom bar (fixture cbssn_army.jpg)"),
}


def get_layout(name: str) -> Layout:
    lay = LAYOUTS[name]
    if not lay.regions or lay.note.startswith("unsupported"):
        raise ValueError(f"layout {name!r} is unsupported")
    return lay
