"""
High-Performance ByteTrack Multi-Object Tracker for Intra-Camera Video Feeds.

References:
- "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- State management with Kalman filtering and two-stage low/high confidence association.
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.detector.yolo_detector import Detection
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.matching import iou_distance, linear_assignment


class TrackState(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """
    Single Track Representation with Kalman Filter state and kinematic trajectory history.
    """
    _count = 0

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        class_id: int = 0,
        class_name: str = "person",
        embedding: Optional[np.ndarray] = None,
        world_coords: Optional[Tuple[float, float]] = None,
        camera_id: str = "CAM_DEFAULT"
    ):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.class_id = class_id
        self.class_name = class_name
        self.camera_id = camera_id
        self.world_coords = world_coords

        self.kalman_filter: Optional[KalmanFilter] = None
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.is_activated: bool = False

        self.track_id: int = 0
        self.state: TrackState = TrackState.NEW

        self.history_embeddings: List[np.ndarray] = []
        if embedding is not None:
            self.history_embeddings.append(embedding)

        self.frame_id: int = 0
        self.start_frame: int = 0
        self.tracklet_len: int = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_counter(cls):
        cls._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """Returns bounding box in [top_left_x, top_left_y, width, height] format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Returns bounding box in [x1, y1, x2, y2] format."""
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Returns the bottom center (contact footprint) of the bounding box for floor projection."""
        box = self.tlbr
        return ((box[0] + box[2]) / 2.0, box[3])

    @property
    def smooth_embedding(self) -> Optional[np.ndarray]:
        """Returns exponentially weighted or averaged ReID feature embedding."""
        if not self.history_embeddings:
            return None
        # Compute normalized mean embedding over recent observations
        mean_emb = np.mean(self.history_embeddings[-10:], axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            return mean_emb / norm
        return mean_emb

    def static_tlwh(self) -> np.ndarray:
        return self._tlwh

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Initializes a new tracklet with a fresh ID and initiates Kalman filter."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Re-associates a recovered lost track with a new detection observation."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len += 1
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.world_coords = new_track.world_coords

        if new_track.history_embeddings:
            self.history_embeddings.append(new_track.history_embeddings[-1])
            if len(self.history_embeddings) > 30:
                self.history_embeddings.pop(0)

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: 'STrack', frame_id: int):
        """Standard update on matched tracklet during continuous tracking."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self.world_coords = new_track.world_coords

        if new_track.history_embeddings:
            self.history_embeddings.append(new_track.history_embeddings[-1])
            if len(self.history_embeddings) > 30:
                self.history_embeddings.pop(0)

    def predict(self):
        """Predicts position for current time step using Kalman filter dynamics."""
        if self.mean is None or self.covariance is None:
            return
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Converts [x, y, w, h] to [center_x, center_y, aspect_ratio w/h, height]."""
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: Tuple[float, float, float, float]) -> np.ndarray:
        """Converts [x1, y1, x2, y2] to [x1, y1, width, height]."""
        x1, y1, x2, y2 = tlbr
        return np.array([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


class BYTETracker:
    """
    Multi-Target ByteTrack Tracker supporting low-confidence association and continuous camera tracking.
    """
    def __init__(
        self,
        camera_id: str = "CAM_01",
        track_thresh: float = 0.45,
        match_thresh: float = 0.80,
        low_thresh: float = 0.10,
        max_time_lost: int = 30,
        min_hits: int = 3
    ):
        self.camera_id = camera_id
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.unconfirmed: List[STrack] = []

    def reset(self):
        """Resets all internal tracker states."""
        self.frame_id = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.unconfirmed.clear()

    def update(
        self,
        detections: List[Detection]
    ) -> List[STrack]:
        """
        Processes single-camera detections for current frame and updates tracklets.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # 1. Parse detection objects and categorize by confidence score
        detections_high: List[STrack] = []
        detections_low: List[STrack] = []

        for det in detections:
            tlwh = STrack.tlbr_to_tlwh(det.bbox)
            track_candidate = STrack(
                tlwh=tlwh,
                score=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                embedding=det.embedding,
                world_coords=det.world_coords,
                camera_id=self.camera_id
            )
            if det.confidence >= self.track_thresh:
                detections_high.append(track_candidate)
            elif det.confidence >= self.low_thresh:
                detections_low.append(track_candidate)

        # 2. Filter unconfirmed and confirmed tracks
        unconfirmed: List[STrack] = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 3. Predict new positions for confirmed and lost tracks using Kalman Filter
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for s in strack_pool:
            s.predict()

        # =====================================================================
        # Stage 1: Match high-confidence detections with existing track pool
        # =====================================================================
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # =====================================================================
        # Stage 2: Match low-confidence detections with remaining unmatched tracks
        # =====================================================================
        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.TRACKED
        ]
        dists = iou_distance(r_tracked_stracks, detections_low)
        matches, u_track_stage2, _ = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track_stage2:
            track = r_tracked_stracks[it]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # =====================================================================
        # Stage 3: Match remaining high-confidence detections with unconfirmed tracks
        # =====================================================================
        detections_high_remaining = [detections_high[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections_high_remaining)
        matches, u_unconfirmed, u_detection_remaining = linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_high_remaining[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # =====================================================================
        # Stage 4: Initialize brand new tracks from unmatched high-confidence detections
        # =====================================================================
        for inew in u_detection_remaining:
            track = detections_high_remaining[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # =====================================================================
        # Stage 5: Clean up dead tracks that exceeded max_time_lost
        # =====================================================================
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.TRACKED]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Combines two track lists while deduplicating by track_id."""
    exists: Dict[int, int] = {}
    res: List[STrack] = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if tid not in exists:
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: List[STrack], tlistb: List[STrack]) -> List[STrack]:
    """Subtracts tracks in tlistb from tlista."""
    stracks: Dict[int, STrack] = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(
    stracksa: List[STrack], stracksb: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Removes duplicate tracks with high spatial overlap (>0.85 IoU)."""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
