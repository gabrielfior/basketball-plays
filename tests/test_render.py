import numpy as np

from basketball_plays import render
from basketball_plays.rosters import DUKE, MICHIGAN
from basketball_plays.schema import PlayerTrack, Possession


def sample_possession():
    traj = [[round(k / 10, 3), 10.0 + k, 25.0] for k in range(20)]
    boxes = [[round(k / 10, 3), 100 + k, 200, 140 + k, 300] for k in range(20)]
    return Possession(
        possession_id=1, start_time=0.0, end_time=2.0, fps=10, offense_team=DUKE,
        attacking_basket="left",
        players=[PlayerTrack(1, DUKE, "12", "Cameron Boozer", traj, boxes),
                 PlayerTrack(2, MICHIGAN, None, None, [[t, x, y + 10] for t, x, y in traj], boxes)],
        ball=[[round(k / 10, 3), 12.0 + k, 26.0] for k in range(20)],
    )


def test_possession_times():
    ts = render.possession_times(sample_possession())
    assert ts[0] == 0.0 and ts[-1] == 1.9 and len(ts) == 20


def test_render_court_frame_shape_and_marks():
    pos = sample_possession()
    img = render.render_court_frame(pos, 1.0, scale=10, padding=30, header=44)
    assert img.shape == (44 + 500 + 60, 940 + 60, 3)
    # a Duke-coloured pixel exists near the player's position (x=20 ft, y=25 ft)
    px, py = 20 * 10 + 30, 25 * 10 + 30 + 44
    region = img[py - 15:py + 15, px - 15:px + 15].reshape(-1, 3)
    assert (region == np.array(render.TEAM_BGR[DUKE])).all(axis=1).any()


def test_draw_overlay_draws_box():
    pos = sample_possession()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = render.draw_overlay(frame, pos, 0.5, render.box_lookup(pos))
    assert out[200, 105].any()  # top edge of the box at x=105
    assert render.label_for(DUKE, "12", "Cameron Boozer", 1) == "DUK #12 Boozer"
    assert render.label_for(MICHIGAN, None, None, 7) == "MIC id7"
    assert render.label_for(MICHIGAN, "21", "Morez Johnson Jr.", 2) == "MIC #21 Johnson"
    assert render.label_for(DUKE, "21", "Patrick Ngongba II", 3) == "DUK #21 Ngongba"


def test_video_writer_roundtrip(tmp_path):
    w = render.VideoWriter(tmp_path / "x.mp4", 10, (64, 32))
    for _ in range(5):
        w.write(np.zeros((32, 64, 3), dtype=np.uint8))
    w.close()
    assert (tmp_path / "x.mp4").stat().st_size > 0


def test_clock_text_and_outcome_header():
    assert render.clock_text(1187) == "19:47"
    assert render.clock_text(45.3) == "45.3"
    pos = sample_possession()
    pos.clock_start, pos.outcome, pos.points_scored = 1187.0, "made_3", 3
    img = render.render_court_frame(pos, 0.5, scale=10, padding=30, header=44)
    assert img.shape[0] == 44 + 500 + 60  # header still one bar


def test_event_banner_and_marker_are_drawn_when_event_is_active():
    pos = sample_possession()
    pos.events = [{"t": 1.0, "clock_text": "19:40", "text": "Cameron Boozer makes 24-foot three point jumper",
                   "type": "JumpShot", "score_value": 3, "player": "Cameron Boozer"},
                  {"t": 0.2, "clock_text": "19:41", "text": "X subbing in for Duke", "type": "Substitution"}]
    assert render.event_style(pos.events[0]) == ("MADE +3", render.GREEN)
    assert render.event_style(pos.events[1]) is None
    assert render.event_style({"text": "Foul on Y.", "type": "PersonalFoul"}) == ("FOUL", render.AMBER)
    assert render.event_style({"text": "Z misses 12-foot jumper", "type": "JumpShot"}) == ("MISSED 2", render.RED)
    assert [e["t"] for e in render.active_events(pos, 1.5)] == [1.0, 0.2]  # the substitution is filtered at draw time
    before = render.render_court_frame(pos, 0.5, scale=10, padding=30, header=44)
    during = render.render_court_frame(pos, 1.5, scale=10, padding=30, header=44)
    assert (before != during).any()
    # green ring around the shooter at x=21 ft (t=1.1 -> k=11)... at t=1.5 the shooter is at x=25
    px, py = 25 * 10 + 30, 25 * 10 + 30 + 44
    ring = during[py - 25:py + 25, px - 25:px + 25].reshape(-1, 3)
    assert (ring == np.array(render.GREEN)).all(axis=1).any()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = render.draw_overlay(frame, pos, 1.5, render.box_lookup(pos))
    assert out[20:40, 20:60].any()  # banner drawn top-left
