"""Vision-based possession segmentation from per-frame court positions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from basketball_plays.court import NCAA, CourtSpec

LEFT = -1
RIGHT = 1


@dataclass
class FrameState:
    frame_idx: int
    t: float
    valid: bool  # court view with a usable homography and enough players
    action_half: int | None  # LEFT / RIGHT, None when unknown


@dataclass
class Segment:
    frame_indices: list[int]  # positions into the FrameState list (valid frames only)
    half: int

    def duration(self, states: list[FrameState], fps: float) -> float:
        return states[self.frame_indices[-1]].t - states[self.frame_indices[0]].t + 1.0 / fps


def action_half(court_xy: np.ndarray, spec: CourtSpec = NCAA) -> int | None:
    """Which half the action is in, from the median x of on-court players."""
    xy = np.asarray(court_xy)
    if xy.size == 0:
        return None
    return LEFT if float(np.median(xy[:, 0])) < spec.length / 2 else RIGHT


def _debounce(labels: list[int], min_run: int) -> list[int]:
    """Relabel runs shorter than min_run to the surrounding label when both neighbours agree."""
    labels = list(labels)
    changed = True
    while changed and labels:
        changed = False
        runs = []  # (label, start, end_exclusive)
        s = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[s]:
                runs.append((labels[s], s, i))
                s = i
        for k, (lab, a, b) in enumerate(runs):
            if b - a >= min_run:
                continue
            prev_lab = runs[k - 1][0] if k > 0 else None
            next_lab = runs[k + 1][0] if k + 1 < len(runs) else None
            target = None
            surrounded = prev_lab is not None and next_lab is not None and prev_lab == next_lab
            if surrounded or (prev_lab is not None and next_lab is None):
                target = prev_lab
            elif prev_lab is None and next_lab is not None:
                target = next_lab
            if target is not None and target != lab:
                labels[a:b] = [target] * (b - a)
                changed = True
                break
    return labels


def segment(
    states: list[FrameState],
    fps: float,
    min_duration: float = 3.0,
    min_flip_duration: float = 1.5,
    max_gap: float = 3.0,
    merge_same_half_gap: float = 12.0,
) -> list[Segment]:
    """Split valid frames into possessions.

    A possession is a maximal run of valid frames on the same half. Half flips shorter than
    `min_flip_duration` are treated as noise. Invalid stretches up to `max_gap` seconds are
    bridged (their frames are excluded); longer ones end the possession, except that two
    consecutive runs on the same half separated by at most `merge_same_half_gap` seconds (a
    replay or close-up in the middle of a possession) are joined. Runs shorter than
    `min_duration` are dropped.
    """
    valid_pos = [i for i, s in enumerate(states) if s.valid and s.action_half is not None]
    if not valid_pos:
        return []
    labels = _debounce([states[i].action_half for i in valid_pos], int(round(min_flip_duration * fps)))

    segments: list[Segment] = []
    cur: list[int] = [valid_pos[0]]
    cur_half = labels[0]
    for k in range(1, len(valid_pos)):
        i = valid_pos[k]
        gap = states[i].t - states[valid_pos[k - 1]].t
        if labels[k] != cur_half or gap > max_gap:
            segments.append(Segment(cur, cur_half))
            cur, cur_half = [i], labels[k]
        else:
            cur.append(i)
    segments.append(Segment(cur, cur_half))
    merged: list[Segment] = []
    for seg in segments:
        if merged and merged[-1].half == seg.half:
            gap = states[seg.frame_indices[0]].t - states[merged[-1].frame_indices[-1]].t
            if gap <= merge_same_half_gap:
                merged[-1].frame_indices += seg.frame_indices
                continue
        merged.append(seg)
    return [s for s in merged if s.duration(states, fps) >= min_duration]


def learn_offense_map(votes: list[tuple[int, int]]) -> dict[int, int]:
    """Map action half -> team cluster on offense, from (half, cluster) votes.

    Votes come from `player-in-possession` detections. Within one period each team attacks a
    fixed basket, so the two halves must map to different clusters; the assignment with the
    most agreeing votes wins.
    """
    counts = {(LEFT, 0): 0, (LEFT, 1): 0, (RIGHT, 0): 0, (RIGHT, 1): 0}
    for half, cluster in votes:
        if (half, cluster) in counts:
            counts[(half, cluster)] += 1
    option_a = counts[(RIGHT, 1)] + counts[(LEFT, 0)]
    option_b = counts[(RIGHT, 0)] + counts[(LEFT, 1)]
    if option_a >= option_b:
        return {RIGHT: 1, LEFT: 0}
    return {RIGHT: 0, LEFT: 1}


def segment_offense(
    votes_by_frame: dict[int, list[int]],
    seg: Segment,
    states: list[FrameState],
    all_segments: list[Segment],
    min_votes: int = 5,
    window_s: float = 300.0,
    global_map: dict[int, int] | None = None,
) -> int | None:
    """Decide which team cluster is on offense for one segment.

    Teams switch baskets at half time, so a whole-game offense map (`learn_offense_map` over
    every vote in the video) can be wrong within a single period. Prefer this segment's own
    `player-in-possession` votes; if there are too few (or they tie), borrow votes from segments
    whose time span lies within `window_s` seconds of this one's; if that is still too few, fall
    back to a caller-supplied whole-game map.
    """
    own_votes = [c for i in seg.frame_indices for c in votes_by_frame.get(i, [])]
    if len(own_votes) >= min_votes:
        ranked = Counter(own_votes).most_common()
        if len(ranked) == 1 or ranked[0][1] != ranked[1][1]:
            return ranked[0][0]

    t0 = states[seg.frame_indices[0]].t
    t1 = states[seg.frame_indices[-1]].t
    nearby_votes: list[tuple[int, int]] = []
    for other in all_segments:
        ot0 = states[other.frame_indices[0]].t
        ot1 = states[other.frame_indices[-1]].t
        if ot0 <= t1 + window_s and ot1 >= t0 - window_s:
            for i in other.frame_indices:
                nearby_votes.extend((other.half, c) for c in votes_by_frame.get(i, []))
    if len(nearby_votes) >= min_votes:
        return learn_offense_map(nearby_votes).get(seg.half)
    return global_map.get(seg.half) if global_map is not None else None


def learn_offense_maps_by_span(
    votes: list[tuple[float, int, int]], spans: list[tuple[float, float]]
) -> list[dict[int, int]]:
    """One `learn_offense_map` result per span, from the votes falling inside it.

    Each vote is `(t, action_half, cluster)`. A span's map uses only the votes with
    `t_lo <= t < t_hi`, so periods (which each have their own fixed basket assignment) are
    learned independently instead of averaging over a whole game that may span several.
    """
    maps = []
    for t_lo, t_hi in spans:
        span_votes = [(half, cluster) for t, half, cluster in votes if t_lo <= t < t_hi]
        maps.append(learn_offense_map(span_votes))
    return maps


def span_index(t: float, spans: list[tuple[float, float]], near_s: float = 30.0) -> int | None:
    """Index of the span containing `t`.

    When `t` falls in the gap between (or outside) the given spans, returns the nearest span by
    boundary distance if that distance is at most `near_s`, else None.
    """
    for i, (lo, hi) in enumerate(spans):
        if lo <= t < hi:
            return i
    best_i, best_d = None, None
    for i, (lo, hi) in enumerate(spans):
        d = min(abs(t - lo), abs(t - hi))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i if best_d is not None and best_d <= near_s else None
