import numpy as np

from basketball_plays import montage as Mo
from basketball_plays.halfcourt import HalfcourtRecord


def rec(players, setup=100.0, handler=None):
    return HalfcourtRecord(
        game_id="g", index=3, team="Duke", start_type="ato", terminal="shot", free_throws=False,
        clock_start=1000, clock_end=990, t_start=99.0, t_end=110.0, t0=99.5, setup=setup,
        no_setup=False, transition=False, located=True, outcome="made_3", points=3,
        n_visible_at_setup=len(players), players=players, ball_handler=handler or [], events=[],
    )


def player(tid, name, jersey, xy_by_t):
    return {"track_id": tid, "name": name, "jersey": jersey,
            "trajectory": [[t, x, y] for t, (x, y) in sorted(xy_by_t.items())]}


def test_frontcourt_image_is_half_the_court():
    img = Mo.frontcourt_image()
    assert img.shape[1] == int(round(47 * Mo.SCALE)) + 2 * Mo.PADDING   # 306 px at SCALE 6
    assert img.shape[0] == int(round(50 * Mo.SCALE)) + 2 * Mo.PADDING


def test_positions_at_picks_the_nearest_frame_and_labels_by_jersey_then_id():
    r = rec([player(1, "Cameron Boozer", "12", {100.0: (20.0, 25.0), 100.1: (21.0, 25.0)}),
             player(2, None, None, {100.0: (28.0, 25.0)})])
    got = Mo.positions_at(r, 100.0)
    assert ("12", 20.0, 25.0) in got and ("2", 28.0, 25.0) in got
    assert Mo.positions_at(r, 105.0) == []


def test_render_setup_tile_has_fixed_size_and_draws_something():
    r = rec([player(i, None, str(i), {100.0 + k / 10: (20.0 + i, 10.0 + 6 * i + k) for k in range(20)})
             for i in range(5)], handler=[[100.0, 20.0, 10.0]])
    tile = Mo.render_setup_tile(r)
    base = Mo.frontcourt_image()
    assert tile.shape == (base.shape[0] + Mo.HEADER, base.shape[1], 3)
    # the header carries text and the court carries player discs: both differ from the bare court
    assert not np.array_equal(tile[Mo.HEADER:], base)


def test_render_setup_tile_tolerates_no_setup_and_no_players():
    r = rec([], setup=None)
    r.no_setup, r.t0 = True, None
    tile = Mo.render_setup_tile(r)
    assert tile.shape[2] == 3


def test_grid_pads_the_last_row():
    tiles = [np.zeros((10, 8, 3), np.uint8) for _ in range(5)]
    g = Mo.grid(tiles, cols=3)
    assert g.shape == (20, 24, 3)
