from pathlib import Path

import cv2
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
    ("ncaa_siena.jpg", "ncaa", 677, 22, 16),        # 11:17, Siena 22, Duke 16
    ("cw_wake.jpg", "cw", 492, 18, 17),             # 8:12, Wake Forest 18 at Duke 17
    ("cbssn_army.jpg", "cbssn", 424, 29, 20),       # 7:04, Duke 29 at Army 20
])
def test_layouts_read_the_fixture_frames(name, layout, clock, away, home):
    frame = cv2.imread(str(FIX / name))
    assert frame is not None and frame.shape[:2] == (720, 1280)
    lay = B.get_layout(layout)
    r = sb.read_frame(frame, 0.0, regions=lay.regions, invert=lay.invert)
    assert (r.clock, r.away, r.home) == (clock, away, home)
