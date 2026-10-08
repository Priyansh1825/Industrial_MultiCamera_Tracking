"""
Deep Re-Identification (Re-ID) Feature Extractor for Factory Personnel and Assets.

Extracts normalized L2 feature embeddings (e.g. 512-dim) from cropped bounding boxes.
Supports ONNX / PyTorch models with fallback to a spatial-color descriptor when external weights are absent.
"""
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np

from src.config_loader import config


class ReIDFeatureExtractor:
    """
    Extracts L2-normalized feature embeddings from object bounding box crops.
    Supports separate models for person and vessel classes.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 512,
        input_size: tuple = (256, 128),  # (height, width)
    ):
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.person_session = None
        self.vessel_session = None

        # Try loading models from config
        reid_cfg = config.get_reid()
        person_model = reid_cfg.person_model
        vessel_model = reid_cfg.vessel_model

        # Check if model files exist
        person_path = Path(person_model) if person_model else None
        vessel_path = Path(vessel_model) if vessel_model else None

        if person_path and person_path.exists():
            self._load_onnx_model(person_path, "person")
        else:
            print(f"[ReID] Person model not found at {person_path}. Using fallback.")

        if vessel_path and vessel_path.exists():
            self._load_onnx_model(vessel_path, "vessel")
        else:
            print(f"[ReID] Vessel model not found at {vessel_path}. Using fallback.")

        # If no model_path provided and config models not found, try the legacy model_path
        if model_path and (self.person_session is None and self.vessel_session is None):
            self._load_onnx_model(Path(model_path), "person")

    def _load_onnx_model(self, path: Path, model_type: str):
        """Attempts to load ONNX runtime inference session."""
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                str(path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            if model_type == "person":
                self.person_session = session
            else:
                self.vessel_session = session
            print(f"[ReID] Successfully loaded {model_type} ONNX model from {path}")
        except Exception as e:
            print(f"[ReID] {model_type} ONNX model load failed ({e}). Using spatial feature extractor.")

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resizes, converts to RGB, and standardizes image crop for model input."""
        h, w = self.input_size
        resized = cv2.resize(crop, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1] and standardize with ImageNet mean/std
        tensor = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (tensor - mean) / std
        # (H, W, C) -> (1, C, H, W)
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def extract_fallback_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extracts a robust multi-zone color moment, circular hue, and spatial descriptor
        when external deep neural network weights are not mounted.
        """
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # 1. Convert to RGB and HSV
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 2. Divide body crop into 3 vertical spatial zones: Head/Hat (top 25%), Torso (40%), Legs/Base (35%)
        h_crop = crop.shape[0]
        zones_rgb = [
            rgb[: int(h_crop * 0.25), :],
            rgb[int(h_crop * 0.25): int(h_crop * 0.65), :],
            rgb[int(h_crop * 0.65):, :]
        ]
        zones_hsv = [
            hsv[: int(h_crop * 0.25), :],
            hsv[int(h_crop * 0.25): int(h_crop * 0.65), :],
            hsv[int(h_crop * 0.65):, :]
        ]

        features = []
        for z_rgb, z_hsv in zip(zones_rgb, zones_hsv):
            if z_rgb.size == 0:
                features.extend([0.0] * 32)
                continue

            # RGB channel statistics
            r_mean, g_mean, b_mean = np.mean(z_rgb, axis=(0, 1))
            # Color Opponency: Red-Green and Blue-Yellow channels
            rg_opp = (r_mean - g_mean)
            by_opp = (b_mean - 0.5 * (r_mean + g_mean))

            # Circular Hue angle (theta in [0, 2*pi])
            h_angles = (z_hsv[:, :, 0] / 180.0) * (2.0 * np.pi)
            cos_h = np.mean(np.cos(h_angles))
            sin_h = np.mean(np.sin(h_angles))
            s_mean = np.mean(z_hsv[:, :, 1]) / 255.0
            v_mean = np.mean(z_hsv[:, :, 2]) / 255.0

            # Weighted circular hue by saturation
            h_vector = np.array([cos_h * s_mean, sin_h * s_mean, rg_opp, by_opp, v_mean], dtype=np.float32)

            # Smooth HSV histogram (8 H bins, 4 S bins)
            h_bins = np.histogram(z_hsv[:, :, 0], bins=8, range=(0, 180))[0].astype(np.float32)
            h_bins /= max(1.0, np.sum(h_bins))

            z_feat = np.concatenate([h_vector * 2.0, h_bins])
            z_norm = np.linalg.norm(z_feat)
            if z_norm > 1e-6:
                z_feat = z_feat / z_norm
            features.append(z_feat)

        # 3. Global feature vector
        combined = np.concatenate(features)

        # Pad or interpolate to target embedding dimension
        if len(combined) < self.embedding_dim:
            # Repeat or pad
            repeats = int(np.ceil(self.embedding_dim / len(combined)))
            extended = np.tile(combined, repeats)[:self.embedding_dim]
        else:
            extended = combined[:self.embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(extended)
        if norm > 1e-6:
            extended = extended / norm
        return extended.astype(np.float32)

    def extract(self, crop: np.ndarray, class_name: str = "person") -> np.ndarray:
        """Extracts single L2-normalized embedding vector from a cropped box."""
        if crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Select appropriate session based on class
        session = self.person_session if class_name == "person" else self.vessel_session

        if session is not None:
            try:
                input_tensor = self.preprocess_crop(crop)
                input_name = session.get_inputs()[0].name
                raw_emb = session.run(None, {input_name: input_tensor})[0].flatten()
                norm = np.linalg.norm(raw_emb)
                if norm > 1e-6:
                    return (raw_emb / norm).astype(np.float32)
                return raw_emb.astype(np.float32)
            except Exception:
                pass

        return self.extract_fallback_embedding(crop)

    def extract_batch(
        self,
        frame: np.ndarray,
        bboxes: List[Union[tuple, list]],
        class_names: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """Extracts embeddings for multiple bounding boxes from a single video frame."""
        h_frame, w_frame = frame.shape[:2]
        embeddings = []
        if class_names is None:
            class_names = ["person"] * len(bboxes)

        for box, cls_name in zip(bboxes, class_names):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w_frame - 1, x1))
            y1 = max(0, min(h_frame - 1, y1))
            x2 = max(x1 + 1, min(w_frame, x2))
            y2 = max(y1 + 1, min(h_frame, y2))

            crop = frame[y1:y2, x1:x2]
            embeddings.append(self.extract(crop, cls_name))
        return embeddings


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Computes cosine similarity between two L2-normalized embeddings in range [-1.0, 1.0]."""
    dot = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_cosine_distance_matrix(gallery_embeddings: np.ndarray, query_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Cosine Distance Matrix (1.0 - CosineSimilarity) between queries and gallery.
    gallery: (N, D)
    query: (M, D)
    Returns: (M, N) distance matrix in [0.0, 2.0]
    """
    if gallery_embeddings.size == 0 or query_embeddings.size == 0:
        return np.empty((len(query_embeddings), len(gallery_embeddings)), dtype=np.float32)

    # Normalize inputs
    g_norms = np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    g_normed = gallery_embeddings / np.maximum(g_norms, 1e-6)
    q_normed = query_embeddings / np.maximum(q_norms, 1e-6)

    similarity_matrix = np.dot(q_normed, g_normed.T)
    cost_matrix = 1.0 - np.clip(similarity_matrix, -1.0, 1.0)
    return cost_matrix
