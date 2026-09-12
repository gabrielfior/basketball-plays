"""Stage A: GPU perception on Modal.

    modal run modal_app.py --video data/duke_michigan_q1.mp4 --game 401817238 --fps 10

`--game` defaults to "default" for legacy (single-game) runs, e.g.

    modal run modal_app.py --video data/duke_michigan_q1.mp4 --end 2135 --fps 10

Uploads the video to a per-game path on a Modal Volume (/vol/games/<game>/...), fits the team
classifier once, then processes 5-minute chunks in parallel. With both rosters'
numbers (`--home-numbers 1,5,23 --away-numbers 2,11 --home-name Duke --away-name Florida`) the
team classifier is fitted supervised on crops whose jersey number the OCR read for exactly one
roster, making cluster 0 the home team and cluster 1 the away team and writing team_map.json
next to the frames; `--no-team-ocr` (or a missing roster) keeps the old unsupervised clustering. `--end` defaults to the video's
duration (via ffprobe), so the whole video is processed unless `--end` is given. Prints an
estimated cost before doing anything remote and refuses to proceed above `--max-cost`. Writes
<out-dir>/frames_XX.jsonl and <out-dir>/meta.json locally (default data/games/<game>/raw).

Chunks resume: a chunk whose frames_XX.jsonl already exists on the volume returns immediately, so
after a failed run the documented retry is the same command with `--skip-upload --skip-fit` (pass
`--force-chunks` to redo chunks that did finish).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

PLAYER_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
KEYPOINT_MODEL_ID = "basketball-court-detection-2/14"
OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
OCR_PROMPT = "Read the number."

VOL_PATH = "/vol"


def game_paths(game: str) -> dict[str, str]:
    base = f"{VOL_PATH}/games/{game}"
    return {
        "base": base,
        "video": f"{base}/video.mp4",
        "classifier": f"{base}/team_classifier.pkl",
        "raw": f"{base}/raw",
    }


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git", "build-essential", "clang")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "inference-gpu[transformers]>=0.50",
        "supervision>=0.25",
        "numpy<2.3",
        "umap-learn>=0.5",
        "scikit-learn>=1.4",
        "pillow",
    )
    .env({"ONNXRUNTIME_EXECUTION_PROVIDERS": "[CUDAExecutionProvider]"})
    .add_local_python_source("basketball_plays")
)
app = modal.App("basketball-plays", image=image)
vol = modal.Volume.from_name("basketball-plays", create_if_missing=True)


def _hf_token() -> str | None:
    """Hugging Face token from the environment or the local HF cache (used for SigLIP downloads)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    cached = Path.home() / ".cache" / "huggingface" / "token"
    return cached.read_text().strip() if cached.exists() else None


secrets = [modal.Secret.from_dotenv()]
if _hf_token():
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": _hf_token()}))


def _read_frames_ffmpeg(path: str, start_s: float, duration_s: float, fps: float, size: tuple[int, int]):
    """Yield BGR frames sampled at exactly `fps` from start_s for duration_s seconds."""
    import subprocess

    import numpy as np

    w, h = size
    cmd = [
        "ffmpeg", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", path,
        "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    nbytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def _video_size(path: str) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _read_numbers(ocr_model, frame, player_boxes, number_boxes) -> list[tuple[int, str]]:
    """OCR every detected number box, attributed to the smallest player box containing it.

    Returns `(player index, text)` pairs; a number box inside no player box is dropped, and an
    OCR failure is non-fatal. Shared by `fit_teams` (to label crops for the supervised team
    classifier) and `process_chunk` (to emit per-track reads), so both read exactly the same
    padded number crop.
    """
    import numpy as np

    out: list[tuple[int, str]] = []
    if not len(player_boxes) or not len(number_boxes):
        return out
    h, w = frame.shape[:2]
    pboxes = np.asarray(player_boxes, dtype=float)
    areas = (pboxes[:, 2] - pboxes[:, 0]) * (pboxes[:, 3] - pboxes[:, 1])
    for box in number_boxes:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        inside = ((pboxes[:, 0] <= cx) & (cx <= pboxes[:, 2])
                  & (pboxes[:, 1] <= cy) & (cy <= pboxes[:, 3]))
        if not inside.any():
            continue
        owner = int(np.argmin(np.where(inside, areas, np.inf)))
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1) - 10), max(0, int(y1) - 10)
        x2, y2 = min(w, int(x2) + 10), min(h, int(y2) + 10)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        try:
            text = ocr_model.infer(frame[y1:y2, x1:x2], prompt=OCR_PROMPT)[0].response
        except Exception:  # noqa: BLE001 - OCR failures are non-fatal
            continue
        out.append((owner, str(text).strip()))
    return out


@app.function(gpu="L4", secrets=secrets, volumes={VOL_PATH: vol}, timeout=3600)
def fit_teams(
    game: str, start_s: float, end_s: float, n_samples: int = 160,
    home_numbers: list[str] = (), away_numbers: list[str] = (), ocr: bool = True,
    home_name: str = "", away_name: str = "",
) -> dict:
    """Fit the per-game team classifier on player crops sampled across the video.

    With both rosters' jersey numbers and `ocr` on, each sampled crop's jersey number is read
    (same OCR model and padded number box `process_chunk` uses) and crops whose number belongs to
    exactly one roster train a supervised classifier, so cluster 0 is the home team and cluster 1
    the away team by construction. Falls back to the unsupervised UMAP+KMeans fit when either
    team ends up with too few labelled crops.
    """
    import cv2
    import numpy as np
    import supervision as sv
    from inference import get_model

    from basketball_plays.schema import CLS_NUMBER, PLAYER_CLASSES
    from basketball_plays.team import TeamClassifier, center_crop_boxes, crop, label_crops_by_jersey

    paths = game_paths(game)
    path = paths["video"]
    os.makedirs(paths["base"], exist_ok=True)
    label = bool(ocr and len(home_numbers) and len(away_numbers))
    model = get_model(model_id=PLAYER_MODEL_ID)
    ocr_model = get_model(model_id=OCR_MODEL_ID) if label else None
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    crops: list = []
    reads: list[str] = []
    for t in np.linspace(start_s, end_s, n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.infer(frame, confidence=0.4, iou_threshold=0.9, class_agnostic_nms=True)[0]
        det = sv.Detections.from_inference(res)
        players = det[np.isin(det.class_id, PLAYER_CLASSES)]
        if not len(players):
            continue
        frame_reads = [""] * len(players)
        if label:
            for owner, text in _read_numbers(ocr_model, frame, players.xyxy,
                                             det[det.class_id == CLS_NUMBER].xyxy):
                frame_reads[owner] = text
        for k, box in enumerate(center_crop_boxes(players.xyxy)):
            c = crop(frame, box)
            if c.shape[0] >= 8 and c.shape[1] >= 8:
                crops.append(c)
                reads.append(frame_reads[k])
    cap.release()

    labels = label_crops_by_jersey(reads, home_numbers, away_numbers) if label else []
    n_labelled = [sum(1 for y in labels if y == 0), sum(1 for y in labels if y == 1)]
    clf = TeamClassifier(device="cuda")
    supervised = clf.fit_supervised(crops, labels) if labels else False
    if not supervised:
        if label:
            print(f"supervised fit skipped: only {n_labelled} labelled crops; clustering instead")
        clf.fit(crops)
    clf.save(paths["classifier"])
    team_map = None
    if supervised and home_name and away_name:
        # Supervised cluster ids are team ids, so Stage B can be told the mapping outright
        # instead of guessing it from brightness and roster agreement.
        team_map = {"0": home_name, "1": away_name}
        with open(f"{paths['base']}/team_map.json", "w") as f:
            json.dump(team_map, f, indent=2)
    vol.commit()
    return {"n_crops": len(crops), "brightness": clf.brightness, "supervised": supervised,
            "n_labelled": n_labelled, "team_map": team_map}


@app.function(gpu="L4", secrets=secrets, volumes={VOL_PATH: vol}, timeout=3600)
def process_chunk(
    game: str, chunk_id: int, start_s: float, end_s: float, fps: float, ocr_every: int = 10,
    force: bool = False,
) -> str:
    import numpy as np
    import supervision as sv
    from inference import get_model

    from basketball_plays.schema import (
        CLS_BALL, CLS_NUMBER, PLAYER_CLASSES, Detection, FrameRecord,
    )
    from basketball_plays.team import TeamClassifier, center_crop_boxes, crop
    from basketball_plays.video import shot_changed

    paths = game_paths(game)
    path = paths["video"]
    # Resume: a chunk that already ran left its output on the volume, so a retry of a run that
    # died part-way only pays for the chunks that are actually missing.
    out_path = f"{paths['raw']}/frames_{chunk_id:02d}.jsonl"
    vol.reload()  # see what earlier runs committed, not this container's stale view
    if not force and os.path.exists(out_path):
        print(f"chunk {chunk_id}: already done, reusing {out_path}")
        return out_path
    player_model = get_model(model_id=PLAYER_MODEL_ID)
    kp_model = get_model(model_id=KEYPOINT_MODEL_ID)
    ocr_model = get_model(model_id=OCR_MODEL_ID)
    clf = TeamClassifier(device="cuda").load(paths["classifier"])

    size = _video_size(path)
    tracker = sv.ByteTrack(frame_rate=int(fps), lost_track_buffer=int(fps * 2))
    shot_counter = 0
    prev_hist = None
    id_base = chunk_id * 1_000_000

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for i, frame in enumerate(_read_frames_ffmpeg(path, start_s, end_s - start_s, fps, size)):
            t = start_s + i / fps
            changed, prev_hist = shot_changed(prev_hist, frame)
            if changed:
                tracker = sv.ByteTrack(frame_rate=int(fps), lost_track_buffer=int(fps * 2))
                shot_counter += 1
            id_offset = id_base + shot_counter * 1000

            res = player_model.infer(frame, confidence=0.4, iou_threshold=0.9)[0]
            det = sv.Detections.from_inference(res)
            players = det[np.isin(det.class_id, PLAYER_CLASSES)]
            players = players.with_nms(threshold=0.7, class_agnostic=True)
            players = tracker.update_with_detections(players)

            team = np.full(len(players), -1)
            if len(players):
                crops = [crop(frame, b) for b in center_crop_boxes(players.xyxy)]
                team = clf.predict(crops)

            dets = []
            for k in range(len(players)):
                tid = players.tracker_id[k] if players.tracker_id is not None else None
                if tid is None:
                    continue
                dets.append(Detection(
                    track_id=int(id_offset + tid), cls=int(players.class_id[k]),
                    conf=round(float(players.confidence[k]), 3),
                    bbox=[round(float(v), 1) for v in players.xyxy[k]],
                    team_cluster=int(team[k]) if team[k] >= 0 else None,
                ))

            balls = det[det.class_id == CLS_BALL]
            ball = None
            if len(balls):
                j = int(np.argmax(balls.confidence))
                ball = [round(float(v), 1) for v in balls.xyxy[j]]

            numbers = []
            if i % ocr_every == 0 and dets:
                nums = det[det.class_id == CLS_NUMBER]
                pboxes = np.array([d.bbox for d in dets])
                numbers = [{"track_id": dets[owner].track_id, "text": text}
                           for owner, text in _read_numbers(ocr_model, frame, pboxes, nums.xyxy)]

            kres = kp_model.infer(frame, confidence=0.3)[0]
            kps = sv.KeyPoints.from_inference(kres)
            if len(kps) and kps.xy.shape[1] > 0:
                xy = kps.xy[0]
                conf = kps.confidence[0] if kps.confidence is not None else np.ones(len(xy))
                keypoints = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                             for (x, y), c in zip(xy, conf)]
            else:
                keypoints = []

            rec = FrameRecord(frame_idx=int(round(t * 60)), t=round(t, 3), keypoints=keypoints,
                              detections=dets, ball=ball, numbers=numbers, shot=changed)
            f.write(rec.to_json() + "\n")
            n += 1
            if n % 200 == 0:
                print(f"chunk {chunk_id}: {n} frames, t={t:.1f}s")
    vol.commit()
    print(f"chunk {chunk_id}: done, {n} frames -> {out_path}")
    return out_path


@app.local_entrypoint()
def main(
    video: str,
    game: str = "default",
    start: float = 0.0,
    end: float = -1.0,
    fps: float = 10.0,
    chunk: float = 300.0,
    out_dir: str = "",
    skip_upload: bool = False,
    skip_fit: bool = False,
    max_cost: float = 10.0,
    home_numbers: str = "",
    away_numbers: str = "",
    home_name: str = "",
    away_name: str = "",
    no_team_ocr: bool = False,
    # 0.09 is a deliberately conservative ceiling: the Michigan game (77.7 video minutes) actually
    # billed $5.00, i.e. $0.064/min. Keep the estimate above the measured rate.
    cost_per_minute: float = 0.09,
    force_chunks: bool = False,
):
    import subprocess

    if end < 0:
        probed = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video]
        ).decode().strip()
        try:
            end = float(probed)
        except ValueError:
            raise SystemExit(
                f"ffprobe could not read a duration from {video} (got {probed!r}); "
                "the file is missing, truncated or not a video") from None
    minutes = (end - start) / 60
    est = minutes * cost_per_minute
    print(f"estimated cost: {minutes:.1f} video minutes x ${cost_per_minute:.2f} = ${est:.2f}")
    if est > max_cost:
        raise SystemExit(
            f"estimate ${est:.2f} exceeds --max-cost {max_cost}; "
            "pass a higher --max-cost to proceed"
        )

    out_dir = out_dir or f"data/games/{game}/raw"
    paths = game_paths(game)
    if not skip_upload:
        print(f"uploading {video} to {paths['video']} ...")
        with vol.batch_upload(force=True) as batch:
            batch.put_file(video, paths["video"][len(VOL_PATH):])
    meta = {"start": start, "end": end, "fps": fps}
    team_map_path = Path(out_dir) / "team_map.json"
    if not skip_fit:
        home_nums = [n for n in home_numbers.split(",") if n.strip()]
        away_nums = [n for n in away_numbers.split(",") if n.strip()]
        use_ocr = bool(home_nums and away_nums and not no_team_ocr)
        # Jersey labelling only keeps the crops whose number is unambiguous, so sample more
        # frames when it is on.
        fit = fit_teams.remote(game, start, end, 400 if use_ocr else 160, home_nums, away_nums,
                               use_ocr, home_name, away_name)
        print("team classifier:", fit)
        meta["cluster_brightness"] = fit["brightness"]
        meta["n_crops"] = fit["n_crops"]
        meta["supervised"] = fit.get("supervised", False)
        meta["n_labelled"] = fit.get("n_labelled", [0, 0])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if fit.get("team_map"):
            team_map_path.write_text(json.dumps(fit["team_map"], indent=2))
            print(f"wrote {team_map_path} (cluster ids are team ids)")
        elif team_map_path.exists():
            # An earlier supervised run's map would mislabel this unsupervised fit's clusters.
            team_map_path.unlink()
            print(f"removed stale {team_map_path} (this fit was unsupervised)")
    else:
        old = Path(out_dir) / "meta.json"
        if old.exists():
            meta["cluster_brightness"] = json.loads(old.read_text()).get("cluster_brightness", {})

    bounds = []
    s = start
    while s < end:
        bounds.append((game, len(bounds), s, min(end, s + chunk), fps, 10, force_chunks))
        s += chunk
    print(f"processing {len(bounds)} chunks ...")
    retry = (f"modal run modal_app.py --video {video} --game {game} --end {end:.0f} "
             "--skip-upload --skip-fit")
    try:
        result_paths = list(process_chunk.starmap(bounds))
    except BaseException:
        print(f"chunk processing failed; finished chunks are kept on the volume, so retry with:"
              f"\n    {retry}")
        raise

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in result_paths:
        rel = p[len(VOL_PATH) + 1:]
        data = b"".join(vol.read_file(rel))
        local = Path(out_dir) / Path(p).name
        local.write_bytes(data)
        print(f"wrote {local} ({len(data) / 1e6:.1f} MB)")
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done")
