"""
Virtual Pinhole CCTV Camera Simulator:
Computes 3D-to-2D perspective projections, homography matrices, frustum culling,
and renders photorealistic industrial CCTV video streams with ground truth.
"""

from dataclasses import dataclass
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .factory_world import AgentType, FactoryWorld, SimulatedAgent


@dataclass
class CameraGroundTruthDetection:
    global_id: str
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    foot_point: Tuple[float, float]          # (u, v)
    world_coords: Tuple[float, float]        # (X, Y) in meters
    occluded: bool = False
    visibility_ratio: float = 1.0


class VirtualCamera:
    """Simulates an industrial CCTV pinhole camera with full 3D projective geometry."""

    def __init__(
        self,
        camera_id: str,
        name: str,
        pos_3d: Tuple[float, float, float],    # (X_c, Y_c, Z_c) in plant meters
        target_3d: Tuple[float, float, float],  # (X_t, Y_t, Z_t) look-at point
        resolution: Tuple[int, int] = (1280, 720),
        fov_deg: float = 70.0,
        adjacent_camera_ids: Optional[List[str]] = None
    ):
        self.camera_id = camera_id
        self.name = name
        self.pos_3d = np.array(pos_3d, dtype=np.float64)
        self.target_3d = np.array(target_3d, dtype=np.float64)
        self.width, self.height = resolution
        self.fov_deg = fov_deg
        self.adjacent_camera_ids = adjacent_camera_ids or []

        self._compute_camera_matrices()
        self._compute_homography_matrix()

    def _compute_camera_matrices(self) -> None:
        """Computes Extrinsics (R, t) and Intrinsics (K) matrices."""
        # 1. Extrinsic Look-At Matrix
        forward = self.target_3d - self.pos_3d
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / norm

        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_world)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        else:
            right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # Rotation matrix (World -> Camera coordinates)
        R = np.vstack([right, -up, forward])
        t = -R @ self.pos_3d

        self.R = R
        self.t = t
        self.extrinsic = np.hstack([R, t.reshape(3, 1)])

        # 2. Intrinsic Camera Matrix (K)
        focal_length = (self.width / 2.0) / math.tan(math.radians(self.fov_deg / 2.0))
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.K = np.array([
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Full 3x4 Projection Matrix P = K * [R | t]
        self.P = self.K @ self.extrinsic

    def _compute_homography_matrix(self) -> None:
        """
        Computes 3x3 Planar Homography matrix mapping floor plane (X, Y, Z=0)
        directly to image coordinates (u, v, 1).
        """
        # Take columns 0, 1, and 3 of 3x4 Projection Matrix P
        H_world_to_img = np.column_stack([self.P[:, 0], self.P[:, 1], self.P[:, 3]])
        self.H_world_to_img = H_world_to_img
        try:
            self.H_img_to_world = np.linalg.inv(H_world_to_img)
        except np.linalg.LinAlgError:
            self.H_img_to_world = np.eye(3)

    def project_world_point_3d(self, X: float, Y: float, Z: float = 0.0) -> Optional[Tuple[float, float, float]]:
        """
        Projects a 3D world coordinate to 2D image plane (u, v) and camera depth.
        Returns (u, v, depth) or None if point is behind camera.
        """
        p_world_h = np.array([X, Y, Z, 1.0], dtype=np.float64)
        p_cam = self.P @ p_world_h

        depth = p_cam[2]
        if depth <= 0.1:  # Behind camera or too close
            return None

        u = p_cam[0] / depth
        v = p_cam[1] / depth
        return (float(u), float(v), float(depth))

    def get_agent_bounding_box(
        self, agent: SimulatedAgent
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Projects 3D bounding prism of the agent onto the 2D camera sensor.
        Returns (x1, y1, x2, y2, foot_u, foot_v) or None if outside FOV.
        """
        hw = agent.width_m / 2.0
        hl = agent.length_m / 2.0
        h = agent.height_m

        corners_3d = [
            (agent.x - hw, agent.y - hl, 0.0),
            (agent.x + hw, agent.y - hl, 0.0),
            (agent.x + hw, agent.y + hl, 0.0),
            (agent.x - hw, agent.y + hl, 0.0),
            (agent.x - hw, agent.y - hl, h),
            (agent.x + hw, agent.y - hl, h),
            (agent.x + hw, agent.y + hl, h),
            (agent.x - hw, agent.y + hl, h),
        ]

        proj_points = []
        for cx, cy, cz in corners_3d:
            res = self.project_world_point_3d(cx, cy, cz)
            if res is None:
                return None
            proj_points.append(res)

        # Foot bottom center point
        foot_res = self.project_world_point_3d(agent.x, agent.y, 0.0)
        if foot_res is None:
            return None
        foot_u, foot_v, _ = foot_res

        us = [p[0] for p in proj_points]
        vs = [p[1] for p in proj_points]

        x1 = max(0.0, min(us))
        y1 = max(0.0, min(vs))
        x2 = min(float(self.width - 1), max(us))
        y2 = min(float(self.height - 1), max(vs))

        # Check if visible inside frame
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        if x1 >= self.width or x2 <= 0 or y1 >= self.height or y2 <= 0:
            return None

        return (x1, y1, x2, y2, foot_u, foot_v)

    def get_ground_truth(self, world: FactoryWorld) -> List[CameraGroundTruthDetection]:
        """Extracts visible detections and their ground truth parameters for this camera."""
        detections = []
        for aid, agent in world.agents.items():
            proj = self.get_agent_bounding_box(agent)
            if proj is not None:
                x1, y1, x2, y2, fu, fv = proj
                detections.append(CameraGroundTruthDetection(
                    global_id=aid,
                    class_id=agent.agent_type.value,
                    class_name=agent.agent_type.name.lower(),
                    bbox=(x1, y1, x2, y2),
                    foot_point=(fu, fv),
                    world_coords=(agent.x, agent.y),
                    occluded=False,
                    visibility_ratio=1.0
                ))
        return detections

    def render_frame(
        self,
        world: FactoryWorld,
        show_ground_truth_overlay: bool = False
    ) -> np.ndarray:
        """Renders an authentic industrial CCTV perspective video frame."""
        frame = np.full((self.height, self.width, 3), (38, 42, 46), dtype=np.uint8)

        # 1. Render Perspective Floor Grid & Walkways
        self._render_floor_grid(frame)
        self._render_zones(frame, world)
        self._render_obstacles(frame, world)

        # 2. Sort agents by distance from camera for correct depth rendering
        sorted_agents = sorted(
            world.agents.values(),
            key=lambda a: np.linalg.norm(np.array([a.x, a.y, 0.0]) - self.pos_3d),
            reverse=True
        )

        for agent in sorted_agents:
            proj = self.get_agent_bounding_box(agent)
            if proj is not None:
                x1, y1, x2, y2, fu, fv = proj
                self._render_agent_entity(frame, agent, x1, y1, x2, y2, fu, fv)

                if show_ground_truth_overlay:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    tag = f"GT:{agent.id} ({agent.x:.1f}m,{agent.y:.1f}m)"
                    cv2.putText(
                        frame,
                        tag,
                        (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

        # 3. Add CCTV Camera OSD Overlay
        self._render_cctv_osd(frame)
        return frame

    def _render_floor_grid(self, frame: np.ndarray) -> None:
        """Draws perspective-transformed ground grid lines."""
        grid_color = (55, 60, 65)
        for y in range(0, 30, 2):
            p1 = self.project_world_point_3d(0.0, float(y), 0.0)
            p2 = self.project_world_point_3d(40.0, float(y), 0.0)
            if p1 and p2:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), grid_color, 1)

        for x in range(0, 42, 2):
            p1 = self.project_world_point_3d(float(x), 0.0, 0.0)
            p2 = self.project_world_point_3d(float(x), 26.0, 0.0)
            if p1 and p2:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), grid_color, 1)

    def _render_zones(self, frame: np.ndarray, world: FactoryWorld) -> None:
        """Draws perspective colored boundary lines and zone markers."""
        for zone in world.zones.values():
            pts_world = [
                (zone.x_min, zone.y_min),
                (zone.x_max, zone.y_min),
                (zone.x_max, zone.y_max),
                (zone.x_min, zone.y_max),
            ]
            proj_pts = []
            for xw, yw in pts_world:
                p = self.project_world_point_3d(xw, yw, 0.0)
                if p:
                    proj_pts.append([int(p[0]), int(p[1])])

            if len(proj_pts) == 4:
                pts_arr = np.array(proj_pts, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts_arr], zone.color_rgb)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.polylines(frame, [pts_arr], True, (200, 200, 100), 1, cv2.LINE_AA)

                cx, cy = zone.center
                p_c = self.project_world_point_3d(cx, cy, 0.0)
                if p_c and 0 <= p_c[0] < self.width and 0 <= p_c[1] < self.height:
                    cv2.putText(
                        frame,
                        zone.name,
                        (int(p_c[0]) - 60, int(p_c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (180, 180, 180),
                        1,
                        cv2.LINE_AA
                    )

    def _render_obstacles(self, frame: np.ndarray, world: FactoryWorld) -> None:
        """Renders 3D machinery blocks and support pillars."""
        for obs in world.obstacles:
            hw = (obs.x_max - obs.x_min) / 2.0
            hl = (obs.y_max - obs.y_min) / 2.0
            cx = (obs.x_min + obs.x_max) / 2.0
            cy = (obs.y_min + obs.y_max) / 2.0
            h = obs.height_m

            corners = [
                (cx - hw, cy - hl, 0.0), (cx + hw, cy - hl, 0.0),
                (cx + hw, cy + hl, 0.0), (cx - hw, cy + hl, 0.0),
                (cx - hw, cy - hl, h), (cx + hw, cy - hl, h),
                (cx + hw, cy + hl, h), (cx - hw, cy + hl, h)
            ]
            pts_2d = [self.project_world_point_3d(x, y, z) for x, y, z in corners]
            if any(p is None for p in pts_2d):
                continue

            pts = [(int(p[0]), int(p[1])) for p in pts_2d]  # type: ignore

            top_poly = np.array([pts[4], pts[5], pts[6], pts[7]], dtype=np.int32)
            cv2.fillPoly(frame, [top_poly], (85, 95, 105))
            cv2.polylines(frame, [top_poly], True, (120, 130, 140), 1)

            front_poly = np.array([pts[0], pts[1], pts[5], pts[4]], dtype=np.int32)
            cv2.fillPoly(frame, [front_poly], (65, 75, 85))
            cv2.polylines(frame, [front_poly], True, (100, 110, 120), 1)

            side_poly = np.array([pts[1], pts[2], pts[6], pts[5]], dtype=np.int32)
            cv2.fillPoly(frame, [side_poly], (50, 60, 70))
            cv2.polylines(frame, [side_poly], True, (80, 90, 100), 1)

            p_lbl = pts_2d[4]
            if p_lbl:
                cv2.putText(
                    frame,
                    obs.name,
                    (int(p_lbl[0]), int(p_lbl[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (150, 160, 170),
                    1,
                    cv2.LINE_AA
                )

    def _render_agent_entity(
        self,
        frame: np.ndarray,
        agent: SimulatedAgent,
        x1: float, y1: float, x2: float, y2: float,
        fu: float, fv: float
    ) -> None:
        """Renders entity visuals (PPE safety gear, steel vessel tanks, or forklifts)."""
        w = int(x2 - x1)
        h = int(y2 - y1)
        ix1, iy1 = int(x1), int(y1)
        ix2, iy2 = int(x2), int(y2)

        shadow_w = max(4, int(w * 0.7))
        shadow_h = max(2, int(w * 0.25))
        cv2.ellipse(
            frame,
            (int(fu), int(fv)),
            (shadow_w, shadow_h),
            0, 0, 360,
            (20, 22, 24),
            -1
        )

        if agent.agent_type == AgentType.PERSON:
            head_h = max(4, int(h * 0.22))
            torso_h = max(6, int(h * 0.45))

            head_cx = ix1 + w // 2
            head_cy = iy1 + head_h // 2
            head_r = max(3, head_h // 2)

            cv2.circle(frame, (head_cx, head_cy), head_r, (0, 230, 255), -1)
            cv2.circle(frame, (head_cx, head_cy), head_r, (30, 30, 30), 1)

            torso_y1 = iy1 + head_h
            torso_y2 = torso_y1 + torso_h
            cv2.rectangle(
                frame,
                (ix1 + int(w * 0.15), torso_y1),
                (ix2 - int(w * 0.15), torso_y2),
                agent.primary_color_bgr,
                -1
            )
            stripe_y = torso_y1 + torso_h // 2
            cv2.line(
                frame,
                (ix1 + int(w * 0.15), stripe_y),
                (ix2 - int(w * 0.15), stripe_y),
                (240, 240, 240),
                max(1, int(torso_h * 0.15))
            )

            cv2.rectangle(
                frame,
                (ix1 + int(w * 0.2), torso_y2),
                (ix2 - int(w * 0.2), iy2),
                agent.secondary_color_bgr,
                -1
            )

        elif agent.agent_type == AgentType.VESSEL_TANK:
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), agent.primary_color_bgr, -1)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (70, 70, 70), 2)
            band_h = max(4, int(h * 0.25))
            band_y1 = iy1 + int(h * 0.35)
            cv2.rectangle(frame, (ix1, band_y1), (ix2, band_y1 + band_h), agent.secondary_color_bgr, -1)
            cv2.putText(
                frame,
                agent.id,
                (ix1 + 2, band_y1 + band_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        elif agent.agent_type == AgentType.FORKLIFT:
            cv2.rectangle(frame, (ix1, iy1 + int(h * 0.4)), (ix2, iy2), agent.primary_color_bgr, -1)
            cv2.rectangle(frame, (ix1 + int(w * 0.2), iy1), (ix2 - int(w * 0.2), iy1 + int(h * 0.4)), (40, 40, 40), 2)
            cv2.circle(frame, (ix1 + w // 2, iy1), max(3, int(w * 0.08)), (0, 165, 255), -1)

    def _render_cctv_osd(self, frame: np.ndarray) -> None:
        """Renders authentic CCTV camera HUD/OSD overlay."""
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        cam_tag = f"{self.camera_id} - {self.name.upper()}"

        cv2.rectangle(frame, (0, 0), (self.width, 32), (15, 15, 15), -1)
        cv2.putText(frame, cam_tag, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (self.width - 240, 16), 5, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (self.width - 225, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            timestamp_str,
            (self.width - 180, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
