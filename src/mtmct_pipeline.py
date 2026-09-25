"""
Industrial Multi-Target Multi-Camera Tracking (MTMCT) End-to-End Orchestrator Pipeline.

Coordinates:
- Multi-Camera Stream Ingestion
- Object Detection (YOLO / Simulated Sensors)
- Intra-Camera Single-Target Tracking (ByteTrack)
- Re-ID Feature Extraction
- Planar Homography Projection (Pixel to Metric Floor Map)
- Spatio-Temporal Graph Matching & Cross-Camera Global ID Association
- Real-Time Spatial Safety Analytics & Zone Monitoring
"""
import time
from typing import Dict, List, Tuple
import numpy as np

from src.detector.yolo_detector import Detection
from src.reid.feature_extractor import ReIDFeatureExtractor
from src.reid.gallery import ReIDGallery
from src.simulator.stream_server import MultiCameraSimulatorServer
from src.spatial_engine.cross_camera_matcher import GlobalTracklet, SpatioTemporalGraphMatcher
from src.spatial_engine.homography import PlanarHomography
from src.spatial_engine.spatial_analytics import SafetyAlert, SpatialAnalyticsEngine
from src.tracker.byte_tracker import BYTETracker, STrack


class MTMCTPipelineEngine:
    """
    Core End-to-End Enterprise MTMCT Pipeline Engine.
    """
    def __init__(
        self,
        fps: int = 25,
        plant_dimensions: Tuple[float, float] = (40.0, 25.0)
    ):
        self.fps = fps
        self.plant_dimensions = plant_dimensions

        # 1. Multi-camera simulator or video ingestion server
        self.simulator = MultiCameraSimulatorServer(fps=fps)

        # 2. Per-camera Single Camera Trackers (ByteTrack)
        self.trackers: Dict[str, BYTETracker] = {}
        for cam_id in self.simulator.cameras.keys():
            self.trackers[cam_id] = BYTETracker(camera_id=cam_id, track_thresh=0.45, match_thresh=0.80)

        # 3. Deep Re-ID feature extractor & central gallery
        self.reid_extractor = ReIDFeatureExtractor(embedding_dim=128)
        self.gallery = ReIDGallery(similarity_threshold=0.68)

        # 4. Spatio-Temporal Graph Optimizer
        self.graph_matcher = SpatioTemporalGraphMatcher(
            gallery=self.gallery,
            max_velocity_mps=3.0,
            appearance_weight=0.60,
            spatial_weight=0.40,
            global_matching_window_sec=120.0
        )

        # 5. Spatial Safety & Dwell Time Analytics Engine
        self.spatial_analytics = SpatialAnalyticsEngine(plant_dimensions=plant_dimensions)

        # 6. Camera homography projectors
        self.homographies: Dict[str, PlanarHomography] = {}
        self._initialize_camera_homographies()

        self.frame_index = 0
        self.is_running = False

    def _initialize_camera_homographies(self):
        """Constructs planar homography for each camera to map image foot contact points to floor meters."""
        for cam_id, cam in self.simulator.cameras.items():
            self.homographies[cam_id] = PlanarHomography(cam.H_img_to_world)

    def step(self) -> Tuple[Dict[str, np.ndarray], Dict[str, GlobalTracklet], List[SafetyAlert]]:
        """
        Executes a single end-to-end synchronized processing step across all camera feeds.
        Returns: (annotated_camera_frames, active_global_tracks, new_safety_alerts)
        """
        self.frame_index += 1
        now = time.time()

        # Step 1: Ingest synchronized multi-camera frames and sensor detections
        frames, cam_ground_truth, _ = self.simulator.step()

        all_camera_tracks: Dict[str, List[STrack]] = {}
        annotated_frames: Dict[str, np.ndarray] = {}

        for cam_id, frame in frames.items():
            cam_dets = cam_ground_truth.get(cam_id, [])

            # Step 2: Convert detections into structured Detection objects with ReID embeddings
            structured_dets: List[Detection] = []
            for d in cam_dets:
                x1, y1, x2, y2 = d.bbox
                crop_y1, crop_y2 = int(max(0, y1)), int(min(frame.shape[0], y2))
                crop_x1, crop_x2 = int(max(0, x1)), int(min(frame.shape[1], x2))
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                emb = self.reid_extractor.extract(crop)

                # Floor projection via camera homography
                foot_u = (x1 + x2) / 2.0
                foot_v = y2
                world_x, world_y = self.homographies[cam_id].image_to_world(foot_u, foot_v)

                # Fallback to 3D raycast ground truth if outside calibration bounds
                out_of_bounds = (
                    world_x <= 0.0
                    or world_x >= self.plant_dimensions[0]
                    or world_y <= 0.0
                    or world_y >= self.plant_dimensions[1]
                )
                if out_of_bounds:
                    world_x, world_y = d.world_coords

                structured_dets.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.92,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    embedding=emb,
                    world_coords=(world_x, world_y)
                ))

            # Step 3: Intra-Camera Tracking (ByteTrack)
            active_stracks = self.trackers[cam_id].update(structured_dets)
            all_camera_tracks[cam_id] = active_stracks

            # Step 4: Spatio-Temporal Graph Matching (Intra Track -> Global ID)
            cam_global_mapping = self.graph_matcher.match_camera_tracklets(
                camera_id=cam_id,
                active_tracks=active_stracks,
                current_timestamp=now
            )

            # Step 5: Render visual overlays on camera feed
            annotated_frame = frame.copy()
            for trk in active_stracks:
                gid = cam_global_mapping.get(trk.track_id, f"TRK-{trk.track_id}")
                box = trk.tlbr
                x1, y1, x2, y2 = [int(v) for v in box]

                import cv2
                # Color code by class
                box_color = (0, 255, 120) if trk.class_name == "person" else (255, 160, 20)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                # Label tag
                label_text = f"[{gid}] {trk.class_name.upper()} ({trk.score:.2f})"
                cv2.putText(
                    annotated_frame, label_text, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame, label_text, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
                )

            annotated_frames[cam_id] = annotated_frame

        # Step 6: Update Spatial Safety & Analytics Engine for all active global tracks
        new_alerts: List[SafetyAlert] = []
        for gid, gt in self.graph_matcher.global_tracks.items():
            if gt.is_active:
                alerts = self.spatial_analytics.update_subject_telemetry(
                    subject_id=gid,
                    label=gt.label,
                    class_name=gt.class_name,
                    world_coords=gt.world_coords,
                    timestamp=now
                )
                new_alerts.extend(alerts)

        self.graph_matcher.prune_inactive_tracks(now, timeout_sec=60.0)

        return annotated_frames, self.graph_matcher.global_tracks, new_alerts
