"""Throwaway probe: verify Modal + Roboflow models load and report their metadata."""
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "git", "build-essential", "clang")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install("inference-gpu[transformers]>=0.50", "supervision>=0.25", "numpy<2.3")
    .env({"ONNXRUNTIME_EXECUTION_PROVIDERS": "[CUDAExecutionProvider]"})
)
app = modal.App("basketball-probe", image=image)


@app.function(gpu="L4", secrets=[modal.Secret.from_dotenv()], timeout=1200)
def probe():
    import time

    import numpy as np
    from inference import get_model

    out = {}
    frame = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
    for mid in ["basketball-player-detection-3-ycjdo/4", "basketball-court-detection-2/14"]:
        t0 = time.time()
        m = get_model(model_id=mid)
        load = time.time() - t0
        r = m.infer(frame, confidence=0.3)[0]
        t0 = time.time()
        for _ in range(5):
            m.infer(frame, confidence=0.3)
        lat = (time.time() - t0) / 5
        out[mid] = {
            "type": type(m).__name__,
            "class_names": getattr(m, "class_names", None),
            "load_s": round(load, 1),
            "latency_ms": round(lat * 1000, 1),
            "result_keys": list(r.model_dump().keys()) if hasattr(r, "model_dump") else str(type(r)),
        }
        if hasattr(r, "predictions") and r.predictions:
            p = r.predictions[0]
            if hasattr(p, "keypoints"):
                out[mid]["n_keypoints"] = len(p.keypoints)
                out[mid]["kp_classes"] = [k.class_name for k in p.keypoints]
    t0 = time.time()
    ocr = get_model(model_id="basketball-jersey-numbers-ocr/3")
    out["ocr"] = {"type": type(ocr).__name__, "load_s": round(time.time() - t0, 1)}
    crop = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    t0 = time.time()
    res = ocr.predict(crop, "Read the number.")
    out["ocr"]["latency_ms"] = round((time.time() - t0) * 1000, 1)
    out["ocr"]["sample"] = str(res)[:200]
    return out


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(probe.remote(), indent=2, default=str))
