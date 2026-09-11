"""Appearance-based team clustering (SigLIP embeddings -> UMAP -> KMeans), after roboflow/sports.

Only imported inside the GPU stage; torch/transformers/umap are not project dependencies.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

SIGLIP_MODEL = "google/siglip-base-patch16-224"


def center_crop_boxes(xyxy: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Shrink boxes around their centre so crops emphasise the jersey."""
    xyxy = np.asarray(xyxy, dtype=float)
    if xyxy.size == 0:
        return xyxy.reshape(0, 4)
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = (xyxy[:, 2] - xyxy[:, 0]) * factor
    h = (xyxy[:, 3] - xyxy[:, 1]) * factor
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def crop(frame: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def mean_brightness(crops: list[np.ndarray]) -> float:
    vals = [float(c.mean()) for c in crops if c.size]
    return float(np.mean(vals)) if vals else 0.0


class TeamClassifier:
    def __init__(self, device: str = "cuda", batch_size: int = 64):
        import torch
        import umap
        from sklearn.cluster import KMeans
        from transformers import AutoProcessor, SiglipVisionModel

        self.device = device
        self.batch_size = batch_size
        self.torch = torch
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        self.reducer = umap.UMAP(n_components=3, random_state=0)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.brightness: dict[int, float] = {}

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        import cv2
        from PIL import Image

        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[i:i + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768))

    def fit(self, crops: list[np.ndarray]) -> None:
        feats = self.embed(crops)
        proj = self.reducer.fit_transform(feats)
        labels = self.kmeans.fit_predict(proj)
        for k in (0, 1):
            self.brightness[k] = mean_brightness([c for c, l in zip(crops, labels) if l == k])

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0,), dtype=int)
        proj = self.reducer.transform(self.embed(crops))
        return self.kmeans.predict(proj)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"reducer": self.reducer, "kmeans": self.kmeans, "brightness": self.brightness}, f)

    def load(self, path: str | Path) -> TeamClassifier:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.reducer, self.kmeans, self.brightness = d["reducer"], d["kmeans"], d["brightness"]
        return self
