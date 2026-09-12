"""Appearance-based team assignment.

Two paths, both producing cluster ids 0/1 per player crop:

* supervised (preferred): a per-game logistic regression on SigLIP embeddings, trained on crops
  whose jersey number the OCR read unambiguously for exactly one of the two rosters. Cluster 0 is
  the home team and cluster 1 the away team by construction.
* unsupervised fallback (after roboflow/sports): SigLIP embeddings -> UMAP(3) -> KMeans(2), whose
  cluster ids are arbitrary and get named downstream by brightness and roster agreement.

Only imported inside the GPU stage; torch/transformers/umap are not project dependencies, and are
imported only when no `embedder` is injected (tests inject one and stay on CPU).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from basketball_plays.identity import normalize_number

SIGLIP_MODEL = "google/siglip-base-patch16-224"
EMBED_DIM = 768


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


def label_crops_by_jersey(
    reads: list[str | None], home_numbers, away_numbers
) -> list[int | None]:
    """Turn per-crop OCR reads into supervision labels: 0 = home, 1 = away, None = unusable.

    A read only labels its crop when the number it normalises to (same digit extraction Stage B
    uses, `identity.normalize_number`) is on exactly one of the two rosters. Numbers worn on both
    teams, numbers on neither, and unreadable text all give None, so the training set contains
    only crops whose team is certain.
    """
    home = {n for n in (normalize_number(x) for x in home_numbers) if n is not None}
    away = {n for n in (normalize_number(x) for x in away_numbers) if n is not None}
    out: list[int | None] = []
    for r in reads:
        n = normalize_number(r)
        if n is None or (n in home) == (n in away):
            out.append(None)
        else:
            out.append(0 if n in home else 1)
    return out


class TeamClassifier:
    def __init__(self, device: str = "cuda", batch_size: int = 64, embedder=None):
        from sklearn.cluster import KMeans

        self.device = device
        self.batch_size = batch_size
        self.embedder = embedder
        self.torch = None
        self.model = None
        self.processor = None
        self.reducer = None
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self.brightness: dict[int, float] = {}
        self.linear = None
        self.supervised = False
        if embedder is None:
            import torch
            import umap
            from transformers import AutoProcessor, SiglipVisionModel

            self.torch = torch
            self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
            self.reducer = umap.UMAP(n_components=3, random_state=0)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if self.embedder is not None:
            feats = np.asarray(self.embedder(crops), dtype=float)
            return feats.reshape(len(crops), -1) if len(crops) else np.zeros((0, EMBED_DIM))

        import cv2
        from PIL import Image

        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                         for c in crops[i:i + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                emb = self.model(**inputs).last_hidden_state.mean(dim=1)
                out.append(emb.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMBED_DIM))

    def fit(self, crops: list[np.ndarray]) -> None:
        feats = self.embed(crops)
        proj = self.reducer.fit_transform(feats)
        labels = self.kmeans.fit_predict(proj)
        for k in (0, 1):
            self.brightness[k] = mean_brightness([c for c, l in zip(crops, labels) if l == k])

    def fit_supervised(
        self, crops: list[np.ndarray], labels: list[int | None], min_per_class: int = 15
    ) -> bool:
        """Fit a logistic regression on the crops whose team the jersey OCR settled.

        Returns False (leaving `linear` None, so the caller falls back to `fit`) when either team
        has fewer than `min_per_class` labelled crops -- too little supervision to trust.
        """
        from sklearn.linear_model import LogisticRegression

        pairs = [(c, int(y)) for c, y in zip(crops, labels) if y in (0, 1)]
        counts = {k: sum(1 for _, y in pairs if y == k) for k in (0, 1)}
        if min(counts.values()) < min_per_class:
            return False
        feats = self.embed([c for c, _ in pairs])
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(feats, np.array([y for _, y in pairs]))
        self.linear = model
        self.supervised = True
        for k in (0, 1):
            self.brightness[k] = mean_brightness([c for c, y in pairs if y == k])
        return True

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        if not len(crops):
            return np.zeros((0,), dtype=int)
        feats = self.embed(crops)
        if self.linear is not None:
            return self.linear.predict(feats).astype(int)
        return self.kmeans.predict(self.reducer.transform(feats))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"reducer": self.reducer, "kmeans": self.kmeans,
                         "brightness": self.brightness, "linear": self.linear,
                         "supervised": self.supervised}, f)

    def load(self, path: str | Path) -> TeamClassifier:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.reducer, self.kmeans, self.brightness = d["reducer"], d["kmeans"], d["brightness"]
        # Pickles written before the supervised path existed have neither key.
        self.linear = d.get("linear")
        self.supervised = bool(d.get("supervised", False))
        return self
