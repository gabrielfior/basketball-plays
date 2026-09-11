"""Setup-snapshot tiles and grids for the play-recognition spike."""

from __future__ import annotations

import cv2
import numpy as np

from basketball_plays import halfcourt as H
from basketball_plays.court import NCAA, draw_court, to_pixel
from basketball_plays.halfcourt import HalfcourtRecord
from basketball_plays.render import TEXT_BGR, clock_text

SCALE = 6.0
PADDING = 12
HEADER = 22
DUKE_BGR = (155, 83, 0)
HANDLER_BGR = (0, 110, 255)
TRAIL_BGR = (90, 90, 90)
START_COLOURS = {
    "ato": (0, 140, 255),
    "dead": (60, 170, 60),
    "live": (200, 120, 0),
    "period": (128, 128, 128),
}


def frontcourt_image() -> np.ndarray:
    """The court drawn at SCALE, cropped to the attacking half (x in [0, 47])."""
    full = draw_court(NCAA, SCALE, PADDING, line_thickness=1)
    right = to_pixel((NCAA.length / 2, 0.0), SCALE, PADDING)[0] + PADDING
    return full[:, :right].copy()


MERGE_FT = 1.0


def labelled_positions_at(rec: HalfcourtRecord, t: float) -> list[tuple[str, float, float]]:
    """(label, x, y) per *merged* player position at `t`.

    Raw tracks are merged the same way `halfcourt.positions_at` merges them, so the number of
    discs drawn matches the record's `n_visible_at_setup`; each merged position takes the label
    of the first raw track within `MERGE_FT` of it (jersey, else the track id mod 1000).
    """
    raw: list[tuple[str, float, float]] = []
    for p in rec.players:
        best = None
        for row in p["trajectory"]:
            if (abs(row[0] - t) < 0.051 and np.isfinite(row[1]) and np.isfinite(row[2])
                    and (best is None or abs(row[0] - t) < abs(best[0] - t))):
                best = row
        if best is not None:
            label = p.get("jersey") or str(p["track_id"] % 1000)
            raw.append((label, float(best[1]), float(best[2])))
    key = round(t, 3)
    tracks = [H.Track(i, None, None, {key: (x, y)}, set())
              for i, (_, x, y) in enumerate(raw)]
    out = []
    for x, y in H.positions_at(tracks, t, MERGE_FT):
        label = next((lbl for lbl, rx, ry in raw
                      if np.hypot(x - rx, y - ry) < MERGE_FT), "?")
        out.append((label, x, y))
    return out


def _trail(rec: HalfcourtRecord, p: dict, t0: float, t1: float) -> list[tuple[int, int]]:
    rows = [
        r for r in p["trajectory"]
        if t0 <= r[0] <= t1 and np.isfinite(r[1]) and np.isfinite(r[2])
    ]
    return [to_pixel((r[1], r[2]), SCALE, PADDING) for r in rows]


def render_setup_tile(rec: HalfcourtRecord, trail_s: float = 2.0) -> np.ndarray:
    court = frontcourt_image()
    t = rec.setup if rec.setup is not None else rec.t0
    if t is not None:
        for p in rec.players:
            pts = _trail(rec, p, t, t + trail_s)
            for k in range(1, len(pts)):
                cv2.line(court, pts[k - 1], pts[k], TRAIL_BGR, 1, cv2.LINE_AA)
        handler = next(
            ((x, y) for tt, x, y in rec.ball_handler
             if abs(tt - t) < 0.051 and np.isfinite(x) and np.isfinite(y)),
            None,
        )
        if handler is not None:
            cx, cy = to_pixel(handler, SCALE, PADDING)
            cv2.circle(court, (cx, cy), int(2.2 * SCALE), HANDLER_BGR, 2, cv2.LINE_AA)
        for label, x, y in labelled_positions_at(rec, t):
            if x >= NCAA.length / 2:
                # off the crop: show it as a hollow disc pinned to the half-court edge
                cy = to_pixel((0.0, y), SCALE, PADDING)[1]
                cx = court.shape[1] - int(1.3 * SCALE) - 2
                cv2.circle(court, (cx, cy), int(1.3 * SCALE), DUKE_BGR, 2, cv2.LINE_AA)
                continue
            cx, cy = to_pixel((x, y), SCALE, PADDING)
            cv2.circle(court, (cx, cy), int(1.3 * SCALE), DUKE_BGR, -1, cv2.LINE_AA)
            cv2.putText(court, label, (cx - 4 * len(label), cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)
    bar = np.full((HEADER, court.shape[1], 3), (245, 245, 245), dtype=np.uint8)
    colour = START_COLOURS.get(rec.start_type, (128, 128, 128))
    cv2.rectangle(bar, (0, 0), (6, HEADER), colour, -1)
    flags = ("T" if rec.transition else "") + ("?" if rec.no_setup else "")
    txt = (f"#{rec.index:02d} {rec.start_type:4s} {clock_text(rec.clock_start)} "
           f"{(rec.outcome or '-')[:9]} n={rec.n_visible_at_setup} {flags}")
    cv2.putText(
        bar, txt, (10, HEADER - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_BGR, 1,
        cv2.LINE_AA,
    )
    return np.vstack([bar, court])


def grid(tiles: list[np.ndarray], cols: int = 6) -> np.ndarray:
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    out = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return out
