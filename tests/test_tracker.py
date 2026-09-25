"""
Unit and Integration Tests for Single-Camera ByteTracker & Kalman Filter.
"""
import numpy as np

from src.detector.yolo_detector import Detection
from src.tracker.byte_tracker import BYTETracker, STrack, TrackState
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.matching import bbox_ious, linear_assignment


def test_kalman_filter_dynamics():
    kf = KalmanFilter()
    # Initial bounding box [center_x, center_y, aspect_ratio, height]
    measurement = np.array([100.0, 100.0, 0.5, 200.0], dtype=np.float32)
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    assert np.isclose(mean[0], 100.0)
    assert np.isclose(mean[1], 100.0)

    # Predict next state
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)

    # Update with new measurement moving right
    new_meas = np.array([105.0, 100.0, 0.5, 200.0], dtype=np.float32)
    upd_mean, upd_cov = kf.update(pred_mean, pred_cov, new_meas)

    # Velocity in x should be positive
    assert upd_mean[4] > 0


def test_bbox_iou_computation():
    boxes_a = np.array([
        [0, 0, 100, 100],
        [50, 50, 150, 150]
    ])
    boxes_b = np.array([
        [0, 0, 100, 100],      # Identical to boxes_a[0]
        [200, 200, 300, 300]   # No overlap
    ])
    ious = bbox_ious(boxes_a, boxes_b)
    assert ious.shape == (2, 2)
    assert np.isclose(ious[0, 0], 1.0)
    assert np.isclose(ious[0, 1], 0.0)


def test_linear_assignment_hungarian():
    cost_matrix = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    matches, u_a, u_b = linear_assignment(cost_matrix, thresh=0.5)
    assert (0, 0) in matches
    assert (1, 1) in matches
    assert len(u_a) == 0
    assert len(u_b) == 0


def test_byte_tracker_lifecycle_and_association():
    STrack.reset_counter()
    tracker = BYTETracker(camera_id="CAM_TEST", track_thresh=0.4, match_thresh=0.8)

    # Frame 1: Single detection at (100, 100, 150, 250)
    det1 = [Detection(
        bbox=(100.0, 100.0, 150.0, 250.0),
        confidence=0.95,
        class_id=0,
        class_name="person",
        world_coords=(5.0, 8.0)
    )]
    tracks_f1 = tracker.update(det1)
    assert len(tracks_f1) == 1
    assert tracks_f1[0].track_id == 1
    assert tracks_f1[0].state == TrackState.TRACKED

    # Frame 2: Subject moves slightly right (105, 100, 155, 250)
    det2 = [Detection(
        bbox=(105.0, 100.0, 155.0, 250.0),
        confidence=0.92,
        class_id=0,
        class_name="person",
        world_coords=(5.2, 8.0)
    )]
    tracks_f2 = tracker.update(det2)
    assert len(tracks_f2) == 1
    # Track ID must be preserved
    assert tracks_f2[0].track_id == 1

    # Frame 3: Temporary occlusion / missed detection
    tracks_f3 = tracker.update([])
    assert len(tracks_f3) == 0  # No active output for that frame
    assert len(tracker.lost_stracks) == 1

    # Frame 4: Subject reappears at (115, 100, 165, 250)
    det4 = [Detection(
        bbox=(115.0, 100.0, 165.0, 250.0),
        confidence=0.88,
        class_id=0,
        class_name="person",
        world_coords=(5.5, 8.0)
    )]
    tracks_f4 = tracker.update(det4)
    assert len(tracks_f4) == 1
    # Re-activated track preserves ID 1
    assert tracks_f4[0].track_id == 1
