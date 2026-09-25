"""
Re-Identification (Re-ID) Module for Personnel and Industrial Asset Matching.
"""
from src.reid.feature_extractor import (
    ReIDFeatureExtractor,
    compute_cosine_distance_matrix,
    compute_cosine_similarity
)
from src.reid.gallery import ReIDGallery, TargetIdentity

__all__ = [
    "ReIDFeatureExtractor",
    "compute_cosine_similarity",
    "compute_cosine_distance_matrix",
    "ReIDGallery",
    "TargetIdentity"
]
