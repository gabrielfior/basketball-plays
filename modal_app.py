"""Stage A: GPU perception on Modal.

    modal run modal_app.py --video data/duke_michigan_q1.mp4 --end 2135 --fps 10

Uploads the video to a Modal Volume, fits the team classifier once, then processes 5-minute
chunks in parallel. Writes data/raw/frames_XX.jsonl and data/raw/meta.json locally.
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
VIDEO_NAME = "video.mp4"
CLASSIFIER_NAME = "team_classifier.pkl"

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
secrets = [modal.Secret.from_dotenv()]


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


@app.function(gpu="L4", secrets=secrets, volumes={VOL_PATH: vol}, timeout=3600)
def fit_teams(start_s: float, end_s: float, n_samples: int = 160) -> dict:
    import cv2
    import numpy as np
    import supervision as sv
    from inference import get_model

    from basketball_plays.schema import PLAYER_CLASSES
    from basketball_plays.team import TeamClassifier, center_crop_boxes, crop

    path = f"{VOL_PATH}/{VIDEO_NAME}"
    model = get_model(model_id=PLAYER_MODEL_ID)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    crops = []
    for t in np.linspace(start_s, end_s, n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * src_fps))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.infer(frame, confidence=0.4, iou_threshold=0.9, class_agnostic_nms=True)[0]
        det = sv.Detections.from_inference(res)
        det = det[np.isin(det.class_id, PLAYER_CLASSES)]
        for box in center_crop_boxes(det.xyxy):
            c = crop(frame, box)
            if c.shape[0] >= 8 and c.shape[1] >= 8:
                crops.append(c)
    cap.release()
    clf = TeamClassifier(device="cuda")
    clf.fit(crops)
    clf.save(f"{VOL_PATH}/{CLASSIFIER_NAME}")
    vol.commit()
    return {"n_crops": len(crops), "brightness": clf.brightness}


@app.function(gpu="L4", secrets=secrets, volumes={VOL_PATH: vol}, timeout=3600)
def process_chunk(chunk_id: int, start_s: float, end_s: float, fps: float, ocr_every: int = 10) -> str:
    import numpy as np
    import supervision as sv
    from inference import get_model

    from basketball_plays.schema import (
        CLS_BALL, CLS_NUMBER, PLAYER_CLASSES, Detection, FrameRecord,
    )
    from basketball_plays.team import TeamClassifier, center_crop_boxes, crop
    from basketball_plays.video import shot_changed

    path = f"{VOL_PATH}/{VIDEO_NAME}"
    player_model = get_model(model_id=PLAYER_MODEL_ID)
    kp_model = get_model(model_id=KEYPOINT_MODEL_ID)
    ocr_model = get_model(model_id=OCR_MODEL_ID)
    clf = TeamClassifier(device="cuda").load(f"{VOL_PATH}/{CLASSIFIER_NAME}")

    size = _video_size(path)
    tracker = sv.ByteTrack(frame_rate=int(fps), lost_track_buffer=int(fps * 2))
    shot_counter = 0
    prev_hist = None
    id_base = chunk_id * 1_000_000

    out_path = f"{VOL_PATH}/raw/frames_{chunk_id:02d}.jsonl"
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
                h, w = frame.shape[:2]
                pboxes = np.array([d.bbox for d in dets])
                areas = (pboxes[:, 2] - pboxes[:, 0]) * (pboxes[:, 3] - pboxes[:, 1])
                for box in nums.xyxy:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    inside = (pboxes[:, 0] <= cx) & (cx <= pboxes[:, 2]) & (pboxes[:, 1] <= cy) & (cy <= pboxes[:, 3])
                    if not inside.any():
                        continue
                    owner = int(np.argmin(np.where(inside, areas, np.inf)))
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, int(x1) - 10), max(0, int(y1) - 10)
                    x2, y2 = min(w, int(x2) + 10), min(h, int(y2) + 10)
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    try:
                        text = ocr_model.predict(frame[y1:y2, x1:x2], OCR_PROMPT)[0]
                    except Exception:  # noqa: BLE001 - OCR failures are non-fatal
                        continue
                    numbers.append({"track_id": dets[owner].track_id, "text": str(text).strip()})

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
    video: str = "data/duke_michigan_q1.mp4",
    start: float = 0.0,
    end: float = 2135.0,
    fps: float = 10.0,
    chunk: float = 300.0,
    out_dir: str = "data/raw",
    skip_upload: bool = False,
    skip_fit: bool = False,
):
    if not skip_upload:
        print(f"uploading {video} to volume ...")
        with vol.batch_upload(force=True) as batch:
            batch.put_file(video, f"/{VIDEO_NAME}")
    meta = {"start": start, "end": end, "fps": fps}
    if not skip_fit:
        fit = fit_teams.remote(start, end)
        print("team classifier:", fit)
        meta["cluster_brightness"] = fit["brightness"]
        meta["n_crops"] = fit["n_crops"]
    else:
        old = Path(out_dir) / "meta.json"
        if old.exists():
            meta["cluster_brightness"] = json.loads(old.read_text()).get("cluster_brightness", {})

    bounds = []
    s = start
    while s < end:
        bounds.append((len(bounds), s, min(end, s + chunk), fps))
        s += chunk
    print(f"processing {len(bounds)} chunks ...")
    paths = list(process_chunk.starmap(bounds))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in paths:
        rel = p[len(VOL_PATH) + 1:]
        data = b"".join(vol.read_file(rel))
        local = Path(out_dir) / Path(p).name
        local.write_bytes(data)
        print(f"wrote {local} ({len(data) / 1e6:.1f} MB)")
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    print("done")
