"""
Spatio-Temporal Graph Optimizer for Multi-Target Multi-Camera Tracking (MTMCT).

Fuses visual Re-ID embeddings with kinematic constraints:
- Velocity bounds (v <= v_max)
- Camera transition topology & temporal window gating
- Multi-objective cost optimization solved via Hungarian bipartite matching.
"""
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.reid.feature_extractor import compute_cosine_similarity
from src.reid.gallery import ReIDGallery
from src.tracker.byte_tracker import STrack


@dataclass
class GlobalTracklet:
    global_id: str
    label: str
    class_name: str
    current_camera: str
    intra_track_id: int
    world_coords: Tuple[float, float]
    bounding_box: Tuple[float, float, float, float]
    embedding: Optional[np.ndarray] = None
    last_updated: float = field(default_factory=time.time)
    trajectory: List[Tuple[float, float, float, str]] = field(default_factory=list)  # (X, Y, timestamp, camera_id)
    is_active: bool = True


class SpatioTemporalGraphMatcher:
    """
    Central MTMCT Cross-Camera Optimizer matching intra-camera tracklets across multiple camera streams.
    """
    def __init__(
        self,
        gallery: ReIDGallery,
        max_velocity_mps: float = 3.0,
        appearance_weight: float = 0.60,
        spatial_weight: float = 0.40,
        global_matching_window_sec: float = 120.0,
        camera_transitions: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    ):
        self.gallery = gallery
        self.max_velocity_mps = max_velocity_mps
        self.w_app = appearance_weight
        self.w_spatial = spatial_weight
        self.matching_window = global_matching_window_sec

        # camera_transitions: {cam_a: {cam_b: (min_travel_sec, max_travel_sec)}}
        self.camera_transitions = camera_transitions or {}
        self.global_tracks: Dict[str, GlobalTracklet] = {}
        self.unmatched_counter = 1

    def compute_kinematic_cost(
        self,
        pos_prev: Tuple[float, float],
        time_prev: float,
        pos_curr: Tuple[float, float],
        time_curr: float,
        cam_prev: str,
        cam_curr: str
    ) -> float:
        """
        Calculates physical plausibility cost based on velocity and camera adjacency.
        Returns: normalized penalty in [0.0, 1.0], or 1e5 if physically impossible.
        """
        dt = max(1e-3, time_curr - time_prev)
        dist = float(np.hypot(pos_curr[0] - pos_prev[0], pos_curr[1] - pos_prev[1]))
        velocity = dist / dt

        # 1. Hard velocity cutoff
        if velocity > self.max_velocity_mps * 1.5:
            return 1e5

        # 2. Check camera transition matrix if crossing cameras
        if cam_prev != cam_curr and cam_prev in self.camera_transitions:
            bounds = self.camera_transitions[cam_prev].get(cam_curr)
            if bounds:
                min_t, max_t = bounds
                if dt < min_t or dt > max_t:
                    # Penalize transition timing anomaly
                    return 0.8 + min(0.2, abs(dt - min_t) / 10.0)

        # 3. Normalized smooth velocity cost (normalized against typical 1.2 m/s walk speed)
        vel_cost = min(1.0, velocity / self.max_velocity_mps)
        return float(vel_cost)

    def match_camera_tracklets(
        self,
        camera_id: str,
        active_tracks: List[STrack],
        current_timestamp: Optional[float] = None
    ) -> Dict[int, str]:
        """
        Matches a camera's active intra-camera STracks to Global Identities.
        Returns mapping: {intra_track_id: global_id}.
        """
        now = current_timestamp or time.time()
        track_to_global: Dict[int, str] = {}

        if not active_tracks:
            return track_to_global

        # 1. Separate tracks that already possess verified associations
        unassigned_tracks: List[STrack] = []
        for track in active_tracks:
            # Check if an active global track already matches this camera & track_id
            matched_gid = None
            for gid, gt in self.global_tracks.items():
                if gt.is_active and gt.current_camera == camera_id and gt.intra_track_id == track.track_id:
                    matched_gid = gid
                    break

            if matched_gid:
                track_to_global[track.track_id] = matched_gid
                # Update existing global track
                w_coords = track.world_coords or (0.0, 0.0)
                gt = self.global_tracks[matched_gid]
                gt.world_coords = w_coords
                gt.bounding_box = tuple(track.tlbr)
                gt.last_updated = now
                gt.trajectory.append((w_coords[0], w_coords[1], now, camera_id))
                if track.smooth_embedding is not None:
                    gt.embedding = track.smooth_embedding
                    self.gallery.update_observation(matched_gid, track.smooth_embedding, camera_id, w_coords, now)
            else:
                unassigned_tracks.append(track)

        if not unassigned_tracks:
            return track_to_global

        # 2. Build candidate pool of global tracks available for cross-camera association
        candidate_global_ids = []
        for gid, gt in self.global_tracks.items():
            # Eligible if not active in the same camera, and updated within the matching window
            time_diff = now - gt.last_updated
            if time_diff <= self.matching_window:
                if not gt.is_active or gt.current_camera != camera_id:
                    candidate_global_ids.append(gid)

        if not candidate_global_ids:
            # No eligible existing tracks -> register unassigned tracks as new global IDs
            for trk in unassigned_tracks:
                gid = self._register_new_track(trk, camera_id, now)
                track_to_global[trk.track_id] = gid
            return track_to_global

        # 3. Form Cost Matrix: [num_unassigned, num_candidates]
        num_u = len(unassigned_tracks)
        num_c = len(candidate_global_ids)
        cost_matrix = np.full((num_u, num_c), 1e5, dtype=np.float32)

        for i, trk in enumerate(unassigned_tracks):
            trk_emb = trk.smooth_embedding
            trk_pos = trk.world_coords or (0.0, 0.0)

            for j, gid in enumerate(candidate_global_ids):
                gt = self.global_tracks[gid]

                # Class consistency check
                if trk.class_name != gt.class_name:
                    continue

                # 1. Appearance Cost (Cosine distance)
                if trk_emb is not None and gt.embedding is not None:
                    sim = compute_cosine_similarity(trk_emb, gt.embedding)
                    app_cost = max(0.0, 1.0 - sim)
                else:
                    app_cost = 0.5  # Neutral appearance cost

                # 2. Kinematic Cost
                kin_cost = self.compute_kinematic_cost(
                    pos_prev=gt.world_coords,
                    time_prev=gt.last_updated,
                    pos_curr=trk_pos,
                    time_curr=now,
                    cam_prev=gt.current_camera,
                    cam_curr=camera_id
                )

                if kin_cost > 100.0:  # Infeasible transition
                    cost_matrix[i, j] = 1e5
                else:
                    cost_matrix[i, j] = self.w_app * app_cost + self.w_spatial * kin_cost

        # 4. Global Bipartite Assignment via Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_unassigned = set()

        for r, c in zip(row_ind, col_ind):
            # Threshold for accepting cross-camera match
            if cost_matrix[r, c] < 0.65:
                trk = unassigned_tracks[r]
                gid = candidate_global_ids[c]
                matched_unassigned.add(r)
                track_to_global[trk.track_id] = gid

                # Re-activate global track in new camera
                w_coords = trk.world_coords or (0.0, 0.0)
                gt = self.global_tracks[gid]
                gt.is_active = True
                gt.current_camera = camera_id
                gt.intra_track_id = trk.track_id
                gt.world_coords = w_coords
                gt.bounding_box = tuple(trk.tlbr)
                gt.last_updated = now
                gt.trajectory.append((w_coords[0], w_coords[1], now, camera_id))
                if trk.smooth_embedding is not None:
                    gt.embedding = trk.smooth_embedding
                    self.gallery.update_observation(gid, trk.smooth_embedding, camera_id, w_coords, now)

        # 5. Register remaining unmatched tracks
        for r, trk in enumerate(unassigned_tracks):
            if r not in matched_unassigned:
                gid = self._register_new_track(trk, camera_id, now)
                track_to_global[trk.track_id] = gid

        return track_to_global

    def _register_new_track(self, trk: STrack, camera_id: str, now: float) -> str:
        """Helper to create a new persistent GlobalTracklet and register in gallery."""
        # Query gallery first to see if it matches a pre-registered known ID (e.g. VSL-402, EMP-108)
        best_id, sim = (None, 0.0)
        if trk.smooth_embedding is not None:
            best_id, sim = self.gallery.query_best_match(trk.smooth_embedding, trk.class_name)

        if best_id and best_id not in self.global_tracks:
            gid = best_id
            label = self.gallery.identities[gid].label
        elif best_id and not self.global_tracks[best_id].is_active:
            gid = best_id
            label = self.global_tracks[best_id].label
        else:
            prefix = "EMP" if trk.class_name == "person" else "VSL"
            gid = f"{prefix}-{self.unmatched_counter:03d}"
            label = f"{trk.class_name.capitalize()} #{self.unmatched_counter}"
            self.unmatched_counter += 1

        w_coords = trk.world_coords or (0.0, 0.0)
        gt = GlobalTracklet(
            global_id=gid,
            label=label,
            class_name=trk.class_name,
            current_camera=camera_id,
            intra_track_id=trk.track_id,
            world_coords=w_coords,
            bounding_box=tuple(trk.tlbr),
            embedding=trk.smooth_embedding,
            last_updated=now,
            trajectory=[(w_coords[0], w_coords[1], now, camera_id)],
            is_active=True
        )
        self.global_tracks[gid] = gt

        if trk.smooth_embedding is not None:
            self.gallery.register_identity(
                global_id=gid,
                label=label,
                class_name=trk.class_name,
                initial_embedding=trk.smooth_embedding,
                camera_id=camera_id,
                world_coords=w_coords
            )
        return gid

    def prune_inactive_tracks(self, current_time: Optional[float] = None, timeout_sec: float = 300.0):
        """Marks long-lost tracks as inactive to prevent unbounded state growth."""
        now = current_time or time.time()
        for gt in self.global_tracks.values():
            if gt.is_active and (now - gt.last_updated) > 2.0:
                gt.is_active = False
