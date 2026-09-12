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
    # When True, `regions` and every entry in `alternatives` are read unconditionally (each
    # alternative merged over `regions`, so it may override only some keys) and the best-scoring
    # clock read wins, rather than only falling back to `alternatives` on parse failure. For a
    # broadcast whose clock shifts position depending on another element's visibility, where both
    # region sets can produce a plausible-but-wrong read and the fallback order can't be trusted.
    compete: bool = False


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
    # narrow blue panel. The boxes were first measured on a first-half frame, where both scores
    # are single digits; on full-game frames the tens digit of a two-digit score fell outside
    # them, so away and home were widened (and trimmed vertically, off the bar's edges) against
    # cw_wake_1500/2500/3500.jpg. All four fixtures now read correctly — two of them only via
    # scoreboard.ocr_digits' glyph-at-a-time fallback, because this bar's condensed italic digits
    # defeat whole-crop OCR at every page-segmentation mode. The clock region's left edge is 985,
    # not 975: at 975 it clips in the tail of the "1ST" label to its left, and with one-digit
    # minutes ("9:54") that sliver becomes its own glyph, misread as a leading "1" ("19:54") —
    # see cw_wake_990.jpg.
    "cw": Layout("cw", {"away": (545, 652, 615, 694), "clock": (985, 652, 1045, 695),
                        "home": (878, 652, 950, 694)},
                 note="The CW bottom bar (fixtures cw_wake.jpg, cw_wake_990/1500/2500/3500.jpg)"),
    # CBS Sports Network: same family as CBS but shifted, with the clock in a light grey panel
    # left of the shot clock. The NHL ticker below the bar stays out of every box. When the shot
    # clock panel is hidden (e.g. under 35s left in the half, so it has nothing to show), the
    # clock panel widens and centres, shifting the digits right far enough that the normal-state
    # region clips the last one ("19:09" -> "19:0"). A single wider box can't cover both states:
    # widened enough to catch the shifted digits, it clips into the shot clock's first digit when
    # the panel is in its normal position. So both region sets are read and compete on the
    # ranking in scoreboard._clock_rank (see fixtures cbssn_army.jpg, cbssn_army_2250.jpg,
    # cbssn_army_3000.jpg, cbssn_army_2100.jpg).
    "cbssn": Layout("cbssn", {"away": (492, 613, 556, 661), "clock": (1038, 618, 1116, 659),
                              "home": (890, 613, 962, 661)},
                    alternatives=({"clock": (1060, 618, 1140, 659)},),
                    compete=True,
                    note="CBS Sports Network bottom bar (fixture cbssn_army.jpg), competing "
                         "against the wider clock box used when the shot clock panel is hidden"),
}


def get_layout(name: str) -> Layout:
    lay = LAYOUTS[name]
    if not lay.regions or lay.note.startswith("unsupported"):
        raise ValueError(f"layout {name!r} is unsupported")
    return lay
