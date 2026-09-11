import numpy as np

from basketball_plays import court


def test_ncaa_dimensions():
    c = court.NCAA
    assert c.length == 94.0
    assert c.width == 50.0
    assert len(c.vertices) == 33


def test_vertex_order_matches_roboflow_layout():
    v = np.array(court.NCAA.vertices)
    # corners
    assert tuple(v[0]) == (0.0, 0.0)
    assert tuple(v[5]) == (0.0, 50.0)
    assert tuple(v[27]) == (94.0, 0.0)
    assert tuple(v[32]) == (94.0, 50.0)
    # baskets on the centre line, 5.25 ft from each baseline
    assert np.allclose(v[6], (5.25, 25.0))
    assert np.allclose(v[26], (94 - 5.25, 25.0))
    # half court
    assert np.allclose(v[16], (47.0, 25.0))
    # paint is 12 ft wide and 19 ft deep
    assert np.allclose(v[2], (0.0, 19.0))
    assert np.allclose(v[3], (0.0, 31.0))
    assert np.allclose(v[9], (19.0, 19.0))
    assert np.allclose(v[11], (19.0, 31.0))
    # top of the three point arc
    assert np.allclose(v[13], (5.25 + court.NCAA.three_point_radius, 25.0))
    # court is symmetric end to end
    mirrored = np.stack([94 - v[:, 0], v[:, 1]], axis=1)
    for a, b in [(0, 27), (1, 28), (2, 29), (7, 24), (9, 21), (12, 18), (13, 19)]:
        assert np.allclose(v[a], mirrored[b]), (a, b)


def test_three_point_straight_section_is_on_the_arc():
    c = court.NCAA
    v = np.array(c.vertices)
    rim = v[6]
    corner_end = v[7]  # where straight section meets the arc
    assert np.isclose(np.linalg.norm(corner_end - rim), c.three_point_radius, atol=0.02)
    assert np.isclose(corner_end[1], c.three_point_sideline_offset)


def test_draw_court_returns_image_with_expected_size():
    img = court.draw_court(scale=10, padding=20)
    assert img.shape == (50 * 10 + 40, 94 * 10 + 40, 3)
    assert img.dtype == np.uint8
    # court lines are white somewhere along the centre line
    x, y = court.to_pixel((47.0, 10.0), scale=10, padding=20)
    assert img[y, x].min() > 200


def test_to_pixel_roundtrip():
    x, y = court.to_pixel((10.0, 5.0), scale=20, padding=50)
    assert (x, y) == (250, 150)
