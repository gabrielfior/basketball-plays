"""NCAA men's court geometry in feet, laid out in the Roboflow court-keypoint vertex order.

Coordinate frame: x along the court length in [0, 94], y across the width in [0, 50].
Vertex 0 is the corner that appears top-left in the canonical broadcast view; y grows
downwards in that view, matching image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtSpec:
    length: float = 94.0
    width: float = 50.0
    three_point_radius: float = 22.146  # 22 ft 1.75 in (FIBA arc adopted by NCAA in 2019)
    three_point_sideline_offset: float = 3.34  # straight section is 40 1/8 in from the sideline
    paint_width: float = 12.0
    paint_length: float = 19.0  # baseline to free-throw line
    center_circle_radius: float = 6.0
    restricted_area_radius: float = 4.0
    rim_offset: float = 5.25  # baseline to rim centre
    hash_mark_offset: float = 28.0  # baseline to sideline hash marks
    hash_mark_length: float = 3.0
    edges: tuple[tuple[int, int], ...] = field(
        default=(
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # left baseline
            (2, 9), (11, 3), (9, 10), (10, 11),  # left paint
            (0, 12), (12, 15), (15, 18), (18, 27),  # top sideline
            (15, 16), (16, 17),  # half court line
            (5, 14), (14, 17), (17, 20), (20, 32),  # bottom sideline
            (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),  # right baseline
            (29, 21), (21, 22), (22, 23), (23, 30),  # right paint
            (1, 7), (4, 8), (28, 24), (31, 25),  # three point straight sections
        )
    )

    @property
    def three_point_straight_length(self) -> float:
        """Distance from the baseline to where the straight 3pt section meets the arc."""
        dy = self.width / 2 - self.three_point_sideline_offset
        return self.rim_offset + math.sqrt(self.three_point_radius**2 - dy**2)

    @property
    def vertices(self) -> list[tuple[float, float]]:
        L, W = self.length, self.width
        mid = W / 2
        p0 = (W - self.paint_width) / 2
        p1 = p0 + self.paint_width
        s = self.three_point_sideline_offset
        straight = self.three_point_straight_length
        rim = self.rim_offset
        r3 = self.three_point_radius
        h = self.hash_mark_offset
        pl = self.paint_length
        v = [
            (0.0, 0.0),  # 00 corner
            (0.0, s),  # 01 3pt line meets baseline
            (0.0, p0),  # 02 paint meets baseline
            (0.0, p1),  # 03
            (0.0, W - s),  # 04
            (0.0, W),  # 05 corner
            (rim, mid),  # 06 rim centre
            (straight, s),  # 07 3pt straight section end
            (straight, W - s),  # 08
            (pl, p0),  # 09 free throw line corners
            (pl, mid),  # 10 free throw line centre
            (pl, p1),  # 11
            (h, 0.0),  # 12 hash mark
            (rim + r3, mid),  # 13 top of the arc
            (h, W),  # 14 hash mark
            (L / 2, 0.0),  # 15 half court
            (L / 2, mid),  # 16 centre
            (L / 2, W),  # 17
            (L - h, 0.0),  # 18
            (L - rim - r3, mid),  # 19
            (L - h, W),  # 20
            (L - pl, p0),  # 21
            (L - pl, mid),  # 22
            (L - pl, p1),  # 23
            (L - straight, s),  # 24
            (L - straight, W - s),  # 25
            (L - rim, mid),  # 26 rim centre
            (L, 0.0),  # 27 corner
            (L, s),  # 28
            (L, p0),  # 29
            (L, p1),  # 30
            (L, W - s),  # 31
            (L, W),  # 32 corner
        ]
        return [(round(x, 3), round(y, 3)) for x, y in v]

    def vertices_array(self) -> np.ndarray:
        return np.array(self.vertices, dtype=np.float32)


NCAA = CourtSpec()

WOOD_BGR = (139, 190, 224)
LINE_BGR = (255, 255, 255)


def to_pixel(xy: tuple[float, float], scale: float, padding: int) -> tuple[int, int]:
    return int(round(xy[0] * scale + padding)), int(round(xy[1] * scale + padding))


def _arc(img, center, radius, start_deg, end_deg, scale, padding, thickness, color=LINE_BGR):
    c = to_pixel(center, scale, padding)
    r = int(round(radius * scale))
    cv2.ellipse(img, c, (r, r), 0, start_deg, end_deg, color, thickness, cv2.LINE_AA)


def draw_court(
    spec: CourtSpec = NCAA,
    scale: float = 10.0,
    padding: int = 30,
    line_thickness: int = 2,
    background_bgr=WOOD_BGR,
) -> np.ndarray:
    """Render the court to a BGR image. Court (x, y) maps to pixel via to_pixel()."""
    h = int(round(spec.width * scale)) + 2 * padding
    w = int(round(spec.length * scale)) + 2 * padding
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = background_bgr
    verts = spec.vertices
    for a, b in spec.edges:
        cv2.line(img, to_pixel(verts[a], scale, padding), to_pixel(verts[b], scale, padding),
                 LINE_BGR, line_thickness, cv2.LINE_AA)

    mid = spec.width / 2
    # centre circle
    _arc(img, (spec.length / 2, mid), spec.center_circle_radius, 0, 360, scale, padding,
         line_thickness)
    for rim_x, sign in ((spec.rim_offset, 1), (spec.length - spec.rim_offset, -1)):
        # three point arc between the two straight-section ends
        dy = mid - spec.three_point_sideline_offset
        half_angle = math.degrees(math.asin(dy / spec.three_point_radius))
        base = 0 if sign == 1 else 180
        _arc(img, (rim_x, mid), spec.three_point_radius, base - half_angle, base + half_angle,
             scale, padding, line_thickness)
        # restricted area arc (open towards the court)
        _arc(img, (rim_x, mid), spec.restricted_area_radius, base - 90, base + 90, scale,
             padding, line_thickness)
        # free throw circle: solid half towards mid court, dashed half towards baseline
        ft_x = spec.paint_length if sign == 1 else spec.length - spec.paint_length
        _arc(img, (ft_x, mid), spec.center_circle_radius, base - 90, base + 90, scale, padding,
             line_thickness)
        for k in range(0, 180, 20):
            _arc(img, (ft_x, mid), spec.center_circle_radius, base + 90 + k, base + 90 + k + 10,
                 scale, padding, line_thickness)
        # rim and backboard
        cv2.circle(img, to_pixel((rim_x, mid), scale, padding), max(2, int(0.75 * scale)),
                   LINE_BGR, line_thickness, cv2.LINE_AA)
        bb_x = rim_x - sign * 1.25
        cv2.line(img, to_pixel((bb_x, mid - 3), scale, padding),
                 to_pixel((bb_x, mid + 3), scale, padding), LINE_BGR, line_thickness + 1)
    # hash marks
    for x in (spec.hash_mark_offset, spec.length - spec.hash_mark_offset):
        for y0, y1 in ((0.0, spec.hash_mark_length), (spec.width - spec.hash_mark_length,
                                                      spec.width)):
            cv2.line(img, to_pixel((x, y0), scale, padding), to_pixel((x, y1), scale, padding),
                     LINE_BGR, line_thickness, cv2.LINE_AA)
    return img
