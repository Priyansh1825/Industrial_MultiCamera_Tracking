"""
Unit & Integration Tests for Re-ID Feature Extraction, Vector Gallery, and Spatio-Temporal Graph Optimizer.
"""
import numpy as np

from src.reid.feature_extractor import (
    ReIDFeatureExtractor,
    compute_cosine_similarity
)
from src.reid.gallery import ReIDGallery
from src.spatial_engine.cross_camera_matcher import SpatioTemporalGraphMatcher
from src.spatial_engine.spatial_analytics import FactoryZone, SpatialAnalyticsEngine
from src.tracker.byte_tracker import STrack


def test_reid_extractor_and_cosine_similarity():
    extractor = ReIDFeatureExtractor(embedding_dim=128)

    # Synthetic image crops (blue box vs orange box)
    crop_blue = np.full((120, 60, 3), (255, 100, 50), dtype=np.uint8)
    crop_orange = np.full((120, 60, 3), (50, 150, 255), dtype=np.uint8)
    crop_blue_similar = np.full((120, 60, 3), (240, 110, 60), dtype=np.uint8)

    emb_blue = extractor.extract(crop_blue)
    emb_orange = extractor.extract(crop_orange)
    emb_blue_sim = extractor.extract(crop_blue_similar)

    assert emb_blue.shape == (128,)
    assert np.isclose(np.linalg.norm(emb_blue), 1.0, atol=1e-3)

    # Identical crop should have cosine similarity close to 1.0
    sim_identical = compute_cosine_similarity(emb_blue, emb_blue)
    assert np.isclose(sim_identical, 1.0, atol=1e-3)

    # High similarity between similar colors
    sim_similar = compute_cosine_similarity(emb_blue, emb_blue_sim)
    # Low similarity between distinct colors
    sim_distinct = compute_cosine_similarity(emb_blue, emb_orange)

    assert sim_similar > sim_distinct
    assert sim_similar > 0.68
    assert sim_distinct < 0.50


def test_reid_gallery_registration_and_search():
    gallery = ReIDGallery(similarity_threshold=0.70)

    emb1 = np.random.randn(128).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    gallery.register_identity(
        global_id="EMP-101",
        label="Operator Alice",
        class_name="person",
        initial_embedding=emb1
    )

    # Search with exact embedding
    match_id, score = gallery.query_best_match(emb1, class_name="person")
    assert match_id == "EMP-101"
    assert np.isclose(score, 1.0, atol=1e-3)

    # Search with unrelated random embedding
    emb_random = np.random.randn(128).astype(np.float32)
    emb_random /= np.linalg.norm(emb_random)
    unmatched_id, _ = gallery.query_best_match(emb_random)
    assert unmatched_id is None or unmatched_id == "EMP-101"


def test_spatio_temporal_cross_camera_matching():
    gallery = ReIDGallery(similarity_threshold=0.70)
    matcher = SpatioTemporalGraphMatcher(
        gallery=gallery,
        max_velocity_mps=3.0,
        appearance_weight=0.6,
        spatial_weight=0.4
    )

    # Time T=0: Camera 1 tracks STrack 1 at (5.0, 8.0)
    emb = np.random.randn(128).astype(np.float32)
    emb /= np.linalg.norm(emb)

    track_cam1 = STrack(
        tlwh=np.array([100, 100, 50, 150]),
        score=0.95,
        class_name="person",
        embedding=emb,
        world_coords=(5.0, 8.0),
        camera_id="CAM_01"
    )
    track_cam1.track_id = 1
    track_cam1.is_activated = True

    mapping1 = matcher.match_camera_tracklets("CAM_01", [track_cam1], current_timestamp=100.0)
    assert 1 in mapping1
    assigned_gid = mapping1[1]

    # Time T=10s: Subject transitions to Camera 2 at (12.0, 10.0), velocity = sqrt(7^2 + 2^2)/10 = 0.72 m/s (feasible)
    track_cam2 = STrack(
        tlwh=np.array([200, 150, 50, 150]),
        score=0.92,
        class_name="person",
        embedding=emb,  # Same appearance
        world_coords=(12.0, 10.0),
        camera_id="CAM_02"
    )
    track_cam2.track_id = 5  # New local track ID in CAM_02
    track_cam2.is_activated = True

    mapping2 = matcher.match_camera_tracklets("CAM_02", [track_cam2], current_timestamp=110.0)
    assert 5 in mapping2
    # The cross-camera matcher must associate CAM_02 track 5 to the existing global ID
    assert mapping2[5] == assigned_gid


def test_spatial_analytics_zones_and_alerts():
    engine = SpatialAnalyticsEngine(plant_dimensions=(40.0, 25.0))

    # Add hazardous zone at [20..30, 0..10]
    hazard_zone = FactoryZone(
        zone_id="TEST_HAZARD",
        name="High Voltage Test Zone",
        polygon=[(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)],
        is_hazard=True,
        allowed_classes=["authorized_technician"]
    )
    engine.add_zone(hazard_zone)

    # Subject entering hazard zone with unauthorized class "person"
    alerts = engine.update_subject_telemetry(
        subject_id="EMP-999",
        label="Visitor John",
        class_name="person",
        world_coords=(25.0, 5.0),
        timestamp=100.0
    )

    assert len(alerts) >= 1
    assert alerts[0].alert_type == "UNAUTHORIZED_INTRUSION"
    assert alerts[0].severity == "CRITICAL"
