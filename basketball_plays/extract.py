"""Stage B: turn per-frame detections (frames.jsonl) into possessions (trajectories.jsonl)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from basketball_plays import geometry, identity, paths, possessions
from basketball_plays.court import NCAA
from basketball_plays.rosters import DUKE, MICHIGAN, ROSTERS
from basketball_plays.schema import (
    CLS_PLAYER_IN_POSSESSION,
    PLAYER_CLASSES,
    Detection,
    FrameRecord,
    PlayerTrack,
    Possession,
)
from basketball_plays.stitch import RawTrack, stitch_tracks


@dataclass
class ProjectedFrame:
    fit: geometry.HomographyFit | None
    players: list[Detection]
    court_xy: np.ndarray  # (n_players, 2)
    ball_xy: np.ndarray | None


def project_frame(
    fr: FrameRecord,
    min_conf: float = 0.5,
    max_reprojection_error: float = 2.0,
    court_margin: float = 3.0,
) -> ProjectedFrame:
    fit = geometry.fit_homography(np.array(fr.keypoints, dtype=np.float32), min_conf=min_conf,
                                  max_reprojection_error=max_reprojection_error)
    players = [d for d in fr.detections if d.cls in PLAYER_CLASSES]
    if fit is None:
        return ProjectedFrame(None, [], np.zeros((0, 2)), None)
    xy = fit.to_court(geometry.bottom_center(np.array([d.bbox for d in players])))
    keep = geometry.inside_court(xy, margin=court_margin)
    players = [d for d, k in zip(players, keep) if k]
    xy = xy[keep]
    ball_xy = None
    if fr.ball is not None:
        b = fit.to_court(geometry.bottom_center(np.array([fr.ball])))
        if geometry.inside_court(b, margin=court_margin)[0]:
            ball_xy = b[0]
    return ProjectedFrame(fit, players, xy, ball_xy)


def build_states(
    frames: list[FrameRecord], projected: list[ProjectedFrame], min_players: int = 6
) -> list[possessions.FrameState]:
    states = []
    for fr, pf in zip(frames, projected):
        valid = pf.fit is not None and len(pf.players) >= min_players
        half = possessions.action_half(pf.court_xy) if valid else None
        states.append(possessions.FrameState(fr.frame_idx, fr.t, valid, half))
    return states


def offense_votes(projected: list[ProjectedFrame], states: list[possessions.FrameState]):
    votes = []
    for pf, st in zip(projected, states):
        if not st.valid or st.action_half is None:
            continue
        for d in pf.players:
            if d.cls == CLS_PLAYER_IN_POSSESSION and d.team_cluster is not None:
                votes.append((st.action_half, int(d.team_cluster)))
    return votes


def track_clusters(projected: list[ProjectedFrame]) -> dict[int, int]:
    """Majority team cluster per track id over the whole clip."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for pf in projected:
        for d in pf.players:
            if d.team_cluster is not None:
                votes[d.track_id][int(d.team_cluster)] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def track_reads(frames: list[FrameRecord]) -> dict[int, list[str]]:
    reads: dict[int, list[str]] = defaultdict(list)
    for fr in frames:
        for n in fr.numbers:
            reads[int(n["track_id"])].append(str(n["text"]))
    return reads


def attach_holding(players: list[PlayerTrack], frames: list[FrameRecord]) -> None:
    """Fill PlayerTrack.holding with the times the detector flagged the track's box as
    player-in-possession. Boxes are matched on (time, rounded bbox) because stitching renames
    track ids."""
    holders: set[tuple[float, tuple[float, ...]]] = set()
    for fr in frames:
        for d in fr.detections:
            if d.cls == CLS_PLAYER_IN_POSSESSION:
                holders.add((round(fr.t, 3), tuple(round(float(v), 1) for v in d.bbox)))
    for p in players:
        p.holding = sorted(
            round(float(row[0]), 3) for row in p.boxes
            if (round(float(row[0]), 3), tuple(round(float(v), 1) for v in row[1:5])) in holders
        )


def resolve_team_names(
    projected: list[ProjectedFrame], frames: list[FrameRecord], cluster_brightness: dict[int, float],
    team_map: dict[int, str] | None = None, home_team: str = DUKE, away_team: str = MICHIGAN,
    rosters: dict | None = None,
) -> dict[int, str]:
    if team_map is not None:
        return team_map
    clusters = track_clusters(projected)
    reads = track_reads(frames)
    by_cluster: dict[int, list[str]] = {0: [], 1: []}
    for tid, rs in reads.items():
        c = clusters.get(tid)
        if c in by_cluster:
            by_cluster[c].extend(rs)
    return identity.name_clusters(cluster_brightness, by_cluster, home_team, away_team,
                                  rosters=rosters)


def build_possessions(
    frames: list[FrameRecord],
    fps: float,
    cluster_brightness: dict[int, float] | None = None,
    team_map: dict[int, str] | None = None,
    min_players: int = 6,
    min_duration: float = 3.0,
    min_flip_duration: float = 1.5,
    max_gap: float = 3.0,
    merge_same_half_gap: float = 12.0,
    min_track_seconds: float = 1.5,
    stitch_gap_s: float = 2.5,
    stitch_dist_ft: float = 6.0,
    rosters: dict | None = None,
    home_team: str = DUKE,
    away_team: str = MICHIGAN,
) -> list[Possession]:
    frames = sorted(frames, key=lambda f: f.t)
    projected = [project_frame(f) for f in frames]
    states = build_states(frames, projected, min_players=min_players)
    segments = possessions.segment(states, fps, min_duration=min_duration,
                                   min_flip_duration=min_flip_duration, max_gap=max_gap,
                                   merge_same_half_gap=merge_same_half_gap)
    offense_map = possessions.learn_offense_map(offense_votes(projected, states))
    names = resolve_team_names(projected, frames, cluster_brightness or {}, team_map,
                               home_team=home_team, away_team=away_team, rosters=rosters)
    clusters = track_clusters(projected)
    reads = track_reads(frames)

    out: list[Possession] = []
    for pid, seg in enumerate(segments):
        idx = seg.frame_indices
        times = np.array([frames[i].t for i in idx])
        per_track: dict[int, dict] = defaultdict(lambda: {"t": [], "xy": [], "box": []})
        ball_t, ball_xy = [], []
        for k, i in enumerate(idx):
            pf = projected[i]
            for d, xy in zip(pf.players, pf.court_xy):
                rec = per_track[d.track_id]
                rec["t"].append(k)
                rec["xy"].append(xy)
                rec["box"].append(d.bbox)
            if pf.ball_xy is not None:
                ball_t.append(k)
                ball_xy.append(pf.ball_xy)

        raw = [
            RawTrack(track_id=tid, frames=rec["t"], xy=[tuple(v) for v in rec["xy"]], boxes=rec["box"],
                     cluster=clusters.get(tid))
            for tid, rec in sorted(per_track.items())
        ]
        if stitch_gap_s > 0:
            raw = stitch_tracks(raw, fps, max_gap_s=stitch_gap_s, max_dist_ft=stitch_dist_ft)

        players: list[PlayerTrack] = []
        candidates: list[dict] = []
        for tr in raw:
            if len(tr.frames) < max(2, int(min_track_seconds * fps)):
                continue
            full = np.full((len(idx), 2), np.nan)
            full[tr.frames] = np.array(tr.xy)
            cleaned = paths.clean_path(full)
            a, b = tr.frames[0], tr.frames[-1]
            team = names.get(tr.cluster) if tr.cluster is not None else None
            member_reads = [r for m in tr.members for r in reads.get(m, [])]
            jersey, votes = identity.jersey_votes(
                member_reads, (rosters or ROSTERS).get(team, {})) if team else (None, 0)
            candidates.append({"id": tr.track_id, "team": team, "jersey": jersey, "votes": votes,
                               "start": float(times[a]), "end": float(times[b])})
            players.append(PlayerTrack(
                track_id=tr.track_id, team=team, jersey=jersey, name=None,
                trajectory=[[round(float(times[k]), 3), round(float(cleaned[k, 0]), 2),
                             round(float(cleaned[k, 1]), 2)] for k in range(a, b + 1)],
                boxes=[[round(float(times[k]), 3)] + [round(float(v), 1) for v in box]
                       for k, box in zip(tr.frames, tr.boxes)],
            ))

        resolved = identity.resolve_conflicts(candidates)
        for pl in players:
            pl.jersey = resolved.get(pl.track_id)
            pl.name = (rosters or ROSTERS).get(pl.team, {}).get(pl.jersey) if pl.team else None
        attach_holding(players, [frames[i] for i in idx])

        ball_out: list[list[float]] = []
        if len(ball_t) >= 2:
            full = np.full((len(idx), 2), np.nan)
            full[ball_t] = np.array(ball_xy)
            cleaned = paths.clean_path(full, min_jump_dist=8.0)
            ball_out = [[round(float(times[k]), 3), round(float(cleaned[k, 0]), 2),
                         round(float(cleaned[k, 1]), 2)] for k in ball_t]

        offense_cluster = offense_map.get(seg.half)
        out.append(Possession(
            possession_id=pid,
            start_time=round(float(times[0]), 3),
            end_time=round(float(times[-1]) + 1.0 / fps, 3),
            fps=fps,
            offense_team=names.get(offense_cluster) if offense_cluster is not None else None,
            attacking_basket="left" if seg.half == possessions.LEFT else "right",
            players=players,
            ball=ball_out,
        ))
    return out


def court_spec():
    return NCAA
