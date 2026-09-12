from pathlib import Path

import cv2
import numpy as np
import pytest

from basketball_plays import broadcasts as B
from basketball_plays import scoreboard as sb

FIX = Path(__file__).parent / "fixtures" / "scoreboards"


def test_espn_layout_matches_the_historical_regions():
    assert B.get_layout("espn").regions == sb.REGIONS


def test_unknown_layout_raises():
    with pytest.raises(KeyError):
        B.get_layout("fox")


@pytest.mark.parametrize("name,layout,clock,away,home", [
    ("espn_louisville.jpg", "espn", 545, 16, 20),   # 9:05, Louisville 16 at Duke 20
    ("espn_kansas.jpg", "espn", 503, 24, 18),       # 8:23, Kansas 24, Duke 18
    ("cbs_fsu.jpg", "cbs", 601, 18, 22),            # 10:01, Duke 18 at Florida State 22
    ("ncaa_bar_siena_1800.jpg", "ncaa", 132, 39, 30),   # 2:12, Siena 39, Duke 30 (bottom bar)
    ("ncaa_bar_siena_3000.jpg", "ncaa", 722, 53, 49),   # 12:02, Siena 53, Duke 49 (bottom bar)
    ("ncaa_siena.jpg", "ncaa", 677, 22, 16),        # 11:17, Siena 22, Duke 16 (stacked box, via
                                                     # the ncaa layout's alternative)
    ("cw_wake.jpg", "cw", 492, 18, 17),             # 8:12, Wake Forest 18 at Duke 17
    ("cw_wake_1500.jpg", "cw", 240, 25, 32),        # 4:00, Wake Forest 25 at Duke 32
    # These two read "1" and "7" -- a dropped tens digit -- under whole-crop OCR in every psm at
    # every scale. They pass via ocr_digits' glyph-at-a-time fallback.
    ("cw_wake_2500.jpg", "cw", 724, 51, 61),        # 12:04, Wake Forest 51 at Duke 61
    ("cw_wake_3500.jpg", "cw", 341, 55, 77),        # 5:41, Wake Forest 55 at Duke 77
    ("cw_wake_990.jpg", "cw", 594, 16, 11),         # 9:54, Wake Forest 16 at Duke 11 —
                                                     # one-digit minutes
    ("cbssn_army.jpg", "cbssn", 424, 29, 20),       # 7:04, Duke 29 at Army 20
    # Shot clock hidden: the clock panel widens and centres, shifting the digits right of the
    # normal-state region. Extracted at T=2250.1 rather than the integer second (see
    # task-4l-report.md): at T=2250.0 the away-score crop's whole-crop OCR ties 2-2 between "01"
    # and "51" and the tie-break happens to keep the wrong one, a pre-existing, JPEG-compression
    # -sensitive artifact in ocr_digits unrelated to this task's clock fix.
    ("cbssn_army_2250.jpg", "cbssn", 1149, 51, 33),  # 19:09, Army 51 at Duke 33 (shot clock
                                                      # hidden; needs the wider alternative region)
    ("cbssn_army_3000.jpg", "cbssn", 751, 70, 36),   # 12:31, Army 70 at Duke 36 (shot clock shown;
                                                      # primary region)
    ("cbssn_army_2100.jpg", "cbssn", 9.2, 48, 30),   # 0:09.2, a sub-minute tenths read
])
def test_layouts_read_the_fixture_frames(name, layout, clock, away, home):
    frame = cv2.imread(str(FIX / name))
    assert frame is not None and frame.shape[:2] == (720, 1280)
    lay = B.get_layout(layout)
    r = sb.read_frame(frame, 0.0, regions=lay.regions, invert=lay.invert,
                      alternatives=lay.alternatives, compete=lay.compete)
    assert (r.clock, r.away, r.home) == (clock, away, home)


def test_clock_rank_orders_colon_reads_over_tenths_over_none():
    assert sb._clock_rank("19:09") > sb._clock_rank("19.0") > sb._clock_rank(None)


@pytest.mark.parametrize("texts,winner", [
    # A full m:ss primary wins outright, even against a longer-looking alternative: a spurious
    # extra digit merged into the alternative crop must not outrank a correct, shorter
    # single-digit-minute clock.
    (["7:04", "10:43"], 0),
    # The shifted-state clip ("19:0" -> tenths-shaped "19.0") is not a full m:ss, so the
    # alternative's clean "19:09" competes and wins.
    (["19.0", "19:09"], 1),
    # Same shape (tenths), more digits wins.
    (["4.5", "45.3"], 1),
    # No primary clock at all: the alternative wins by default.
    ([None, "12:31"], 1),
])
def test_pick_clock_index_guards_a_full_primary_then_ranks(texts, winner):
    assert sb._pick_clock_index(texts) == winner


def test_read_frame_returns_none_clock_when_no_alternative_parses():
    """A blank frame: neither the ncaa layout's primary regions nor its alternative parse."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    lay = B.get_layout("ncaa")
    r = sb.read_frame(frame, 0.0, regions=lay.regions, invert=lay.invert,
                      alternatives=lay.alternatives)
    assert r.clock is None
