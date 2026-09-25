"""
Vector Identity Gallery and Long-Term ReID Feature Memory.

Maintains historical feature banks, smooth running centroids, and provides fast vector search.
"""
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.reid.feature_extractor import compute_cosine_distance_matrix, compute_cosine_similarity


@dataclass
class TargetIdentity:
    global_id: str
    label: str
    class_name: str
    feature_bank: List[np.ndarray] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    last_seen_camera: str = ""
    last_seen_world_coords: Tuple[float, float] = (0.0, 0.0)
    last_seen_timestamp: float = field(default_factory=time.time)
    registered_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def add_feature(self, embedding: np.ndarray, max_bank_size: int = 50):
        """Adds a new observed embedding and updates the running unit-norm centroid."""
        if embedding is None or embedding.size == 0:
            return
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            normalized_emb = embedding / norm
        else:
            normalized_emb = embedding

        self.feature_bank.append(normalized_emb)
        if len(self.feature_bank) > max_bank_size:
            self.feature_bank.pop(0)

        # Update centroid via weighted mean
        mean_feat = np.mean(self.feature_bank, axis=0)
        c_norm = np.linalg.norm(mean_feat)
        if c_norm > 1e-6:
            self.centroid = mean_feat / c_norm
        else:
            self.centroid = mean_feat


class ReIDGallery:
    """
    Central Vector Gallery managing global identities across all factory cameras.
    """
    def __init__(self, similarity_threshold: float = 0.70, max_bank_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_bank_size = max_bank_size
        self.identities: Dict[str, TargetIdentity] = {}

    def register_identity(
        self,
        global_id: str,
        label: str,
        class_name: str,
        initial_embedding: Optional[np.ndarray] = None,
        camera_id: str = "",
        world_coords: Tuple[float, float] = (0.0, 0.0),
        metadata: Optional[Dict] = None
    ) -> TargetIdentity:
        """Registers a new identity into the central gallery."""
        target = TargetIdentity(
            global_id=global_id,
            label=label,
            class_name=class_name,
            last_seen_camera=camera_id,
            last_seen_world_coords=world_coords,
            last_seen_timestamp=time.time(),
            metadata=metadata or {}
        )
        if initial_embedding is not None:
            target.add_feature(initial_embedding, self.max_bank_size)
        self.identities[global_id] = target
        return target

    def update_observation(
        self,
        global_id: str,
        embedding: np.ndarray,
        camera_id: str,
        world_coords: Tuple[float, float],
        timestamp: Optional[float] = None
    ):
        """Updates spatial position and visual appearance of a recognized identity."""
        if global_id in self.identities:
            target = self.identities[global_id]
            target.add_feature(embedding, self.max_bank_size)
            target.last_seen_camera = camera_id
            target.last_seen_world_coords = world_coords
            target.last_seen_timestamp = timestamp or time.time()

    def query_best_match(
        self,
        embedding: np.ndarray,
        class_name: Optional[str] = None
    ) -> Tuple[Optional[str], float]:
        """
        Finds the closest registered identity in the gallery by cosine similarity.
        Returns: (best_global_id, similarity_score) or (None, 0.0) if below threshold.
        """
        if embedding is None or not self.identities:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        for gid, identity in self.identities.items():
            if class_name is not None and identity.class_name != class_name:
                continue
            if identity.centroid is None:
                continue

            sim = compute_cosine_similarity(embedding, identity.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = gid

        if best_sim >= self.similarity_threshold:
            return best_id, best_sim
        return None, best_sim

    def compute_distance_matrix(
        self,
        query_embeddings: List[np.ndarray],
        filter_class: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Computes distance matrix between a batch of query embeddings and all gallery centroids.
        Returns: (cost_matrix of shape [num_queries, num_gallery], list_of_gallery_ids)
        """
        gallery_ids = []
        gallery_centroids = []

        for gid, identity in self.identities.items():
            if filter_class is not None and identity.class_name != filter_class:
                continue
            if identity.centroid is not None:
                gallery_ids.append(gid)
                gallery_centroids.append(identity.centroid)

        if not gallery_centroids or not query_embeddings:
            return np.empty((len(query_embeddings), 0), dtype=np.float32), []

        gallery_matrix = np.array(gallery_centroids, dtype=np.float32)
        query_matrix = np.array(query_embeddings, dtype=np.float32)

        cost_matrix = compute_cosine_distance_matrix(gallery_matrix, query_matrix)
        return cost_matrix, gallery_ids
