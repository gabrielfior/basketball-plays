import numpy as np

from basketball_plays import zones as Z


def test_mirror_right_basket_flips_x_only():
    xy = np.array([[10.0, 5.0], [80.0, 45.0]])
    out = Z.mirror_to_canonical(xy, "right")
    assert np.allclose(out, [[84.0, 5.0], [14.0, 45.0]])
    assert np.allclose(Z.mirror_to_canonical(xy, "left"), xy)
    assert out is not xy


def test_named_spots_land_in_expected_zones():
    cases = {
        (5.25, 25.0): "rim",
        (5.25, 20.0): "dunker_far",         # 5 ft from the rim along the baseline, far side
        (5.25, 30.0): "dunker_near",
        (15.0, 18.0): "post_far",           # 12 ft from the rim, 36 degrees to the far side
        (23.0, 25.0): "nail",               # 17.75 ft from the rim, ring 2 centre sector
        (21.0, 13.0): "elbow_far",          # ~19.9 ft, 45 degrees far side
        (28.0, 25.0): "top",                # 22.75 ft, straight on
        (2.0, 2.0): "corner_far",           # behind the rim line folds into the baseline sector
        (2.0, 48.0): "corner_near",
        (22.0, 8.0): "wing_far",            # ~23.9 ft, far wing
        (40.0, 25.0): "deep",
        (50.0, 25.0): "backcourt",
    }
    for (x, y), name in cases.items():
        assert Z.ZONE_NAMES[Z.zone_of(x, y)] == name, (x, y, Z.ZONE_NAMES[Z.zone_of(x, y)])


def test_free_throw_line_centre_is_paint_zone():
    # 19 - 5.25 = 13.75 ft from the rim: second ring, centre sector
    assert Z.ZONE_NAMES[Z.zone_of(19.0, 25.0)] == "paint"


def test_zones_of_and_occupancy_are_consistent():
    xy = np.array([[5.25, 25.0], [28.0, 25.0], [28.0, 25.0], [50.0, 10.0]])
    ids = Z.zones_of(xy)
    assert ids.tolist() == [Z.zone_of(*p) for p in xy.tolist()]
    occ = Z.occupancy(xy)
    assert occ.shape == (Z.N_ZONES,)
    assert occ[Z.ZONE_NAMES.index("top")] == 2
    assert occ[Z.BACKCOURT] == 1
    assert occ.sum() == 4


def test_occupancy_ignores_nan_rows():
    xy = np.array([[np.nan, np.nan], [28.0, 25.0]])
    assert Z.occupancy(xy).sum() == 1


def test_rim_distance():
    assert np.allclose(Z.rim_distance(np.array([[5.25, 25.0], [27.396, 25.0]])), [0.0, 22.146])
