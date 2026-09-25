"""
Multi-Camera Stream Server & Digital Twin Engine.
Coordinates synchronous multi-view video synthesis, bird's-eye 2D spatial rendering,
and real-time ground-truth broadcasting.
"""

import math
import time
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from .factory_world import AgentType, FactoryWorld
from .virtual_camera import CameraGroundTruthDetection, VirtualCamera


class MultiCameraSimulatorServer:
    """Manages the virtual factory floor and synchronized multi-camera CCTV network."""

    def __init__(
        self,
        world: Optional[FactoryWorld] = None,
        cameras: Optional[List[VirtualCamera]] = None,
        fps: int = 25
    ):
        self.world = world or FactoryWorld()
        if world is None:
            self.world.spawn_default_fleet()

        self.cameras: Dict[str, VirtualCamera] = {}
        if cameras:
            for cam in cameras:
                self.cameras[cam.camera_id] = cam
        else:
            self._setup_default_camera_network()

        self.fps = fps
        self.dt = 1.0 / float(fps)
        self.is_running = False

    def _setup_default_camera_network(self) -> None:
        """Configures standard 4-camera industrial CCTV topology with strategic overlap zones."""
        default_cams = [
            VirtualCamera(
                camera_id="CAM_01_GATE_NORTH",
                name="Main North Entrance Gate",
                pos_3d=(2.0, 1.5, 5.5),
                target_3d=(7.5, 8.0, 0.0),
                resolution=(1280, 720),
                fov_deg=75.0,
                adjacent_camera_ids=["CAM_02_AISLE_WEST", "CAM_03_REACTOR_BAY"]
            ),
            VirtualCamera(
                camera_id="CAM_02_AISLE_WEST",
                name="Aisle West Processing Corridor",
                pos_3d=(2.0, 23.5, 5.8),
                target_3d=(7.0, 15.0, 0.0),
                resolution=(1280, 720),
                fov_deg=72.0,
                adjacent_camera_ids=["CAM_01_GATE_NORTH", "CAM_04_LOADING_DOCK"]
            ),
            VirtualCamera(
                camera_id="CAM_03_REACTOR_BAY",
                name="Chemical Reactor & Agitation Bay",
                pos_3d=(26.0, 2.0, 6.2),
                target_3d=(20.0, 9.0, 0.0),
                resolution=(1280, 720),
                fov_deg=74.0,
                adjacent_camera_ids=["CAM_01_GATE_NORTH", "CAM_04_LOADING_DOCK"]
            ),
            VirtualCamera(
                camera_id="CAM_04_LOADING_DOCK",
                name="South Loading & Logistics Dock",
                pos_3d=(38.0, 23.5, 6.0),
                target_3d=(32.0, 13.0, 0.0),
                resolution=(1280, 720),
                fov_deg=75.0,
                adjacent_camera_ids=["CAM_02_AISLE_WEST", "CAM_03_REACTOR_BAY"]
            ),
        ]
        for cam in default_cams:
            self.cameras[cam.camera_id] = cam

    def step(self) -> Tuple[Dict[str, np.ndarray], Dict[str, List[CameraGroundTruthDetection]], Dict[str, dict]]:
        """
        Advances the simulation by one frame interval (dt).
        Returns:
            frames: {cam_id: np.ndarray image}
            cam_detections: {cam_id: list of CameraGroundTruthDetection}
            world_ground_truth: {agent_id: dict ground truth state}
        """
        self.world.step(self.dt)

        frames: Dict[str, np.ndarray] = {}
        cam_detections: Dict[str, List[CameraGroundTruthDetection]] = {}

        for cam_id, cam in self.cameras.items():
            frame = cam.render_frame(self.world, show_ground_truth_overlay=True)
            dets = cam.get_ground_truth(self.world)
            frames[cam_id] = frame
            cam_detections[cam_id] = dets

        world_gt = self.world.get_ground_truth()
        return frames, cam_detections, world_gt

    def render_bird_eye_view(self, map_width: int = 800, map_height: int = 500) -> np.ndarray:
        """
        Renders an authentic 2D Factory Floor Bird's-Eye View (BEV) Digital Twin.
        Includes zone bounds, obstacles, camera positions + FOV vision cones,
        agent positions, and trajectory breadcrumbs.
        """
        bev = np.full((map_height, map_width, 3), (25, 28, 32), dtype=np.uint8)

        # Scale factors (meters -> pixel)
        scale_x = (map_width - 40) / self.world.width_m
        scale_y = (map_height - 40) / self.world.length_m
        offset_x = 20
        offset_y = 20

        def to_px(xm: float, ym: float) -> Tuple[int, int]:
            px = int(offset_x + xm * scale_x)
            py = int(offset_y + ym * scale_y)
            return (px, py)

        # 1. Floor Border & Grid
        cv2.rectangle(
            bev,
            to_px(0.0, 0.0),
            to_px(self.world.width_m, self.world.length_m),
            (70, 75, 80),
            2
        )

        # 2. Zones
        for zone in self.world.zones.values():
            p1 = to_px(zone.x_min, zone.y_min)
            p2 = to_px(zone.x_max, zone.y_max)
            overlay = bev.copy()
            cv2.rectangle(overlay, p1, p2, zone.color_rgb, -1)
            cv2.addWeighted(overlay, 0.35, bev, 0.65, 0, bev)
            cv2.rectangle(bev, p1, p2, (120, 130, 140), 1)

            # Zone name
            cv2.putText(
                bev,
                zone.name,
                (p1[0] + 6, p1[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        # 3. Obstacles
        for obs in self.world.obstacles:
            p1 = to_px(obs.x_min, obs.y_min)
            p2 = to_px(obs.x_max, obs.y_max)
            cv2.rectangle(bev, p1, p2, (80, 85, 95), -1)
            cv2.rectangle(bev, p1, p2, (140, 145, 155), 1)
            cv2.putText(
                bev,
                obs.name,
                (p1[0] + 4, p1[1] + (p2[1] - p1[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

        # 4. Camera FOV Cones
        for cam in self.cameras.values():
            c_pos = to_px(cam.pos_3d[0], cam.pos_3d[1])
            # Draw camera mount icon
            cv2.circle(bev, c_pos, 7, (0, 255, 255), -1)
            cv2.circle(bev, c_pos, 7, (0, 0, 0), 2)
            cv2.putText(
                bev,
                cam.camera_id[:6],
                (c_pos[0] + 10, c_pos[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Draw vision frustum polygon on floor
            t_pos = to_px(cam.target_3d[0], cam.target_3d[1])
            cv2.line(bev, c_pos, t_pos, (0, 200, 200), 1, cv2.LINE_AA)

        # 5. Dynamic Agents & Trajectory Trails
        for agent in self.world.agents.values():
            # Draw trajectory path
            if len(agent.history) > 1:
                trail_pts = [to_px(hx, hy) for hx, hy, _ in agent.history]
                for i in range(1, len(trail_pts)):
                    alpha = i / len(trail_pts)
                    color = (int(0 * alpha), int(215 * alpha), int(255 * alpha))
                    cv2.line(bev, trail_pts[i - 1], trail_pts[i], color, 1, cv2.LINE_AA)

            # Current position
            pos_px = to_px(agent.x, agent.y)

            # Distinct shapes based on class
            if agent.agent_type == AgentType.PERSON:
                cv2.circle(bev, pos_px, 6, agent.primary_color_bgr, -1)
                cv2.circle(bev, pos_px, 6, (255, 255, 255), 1)
            elif agent.agent_type == AgentType.VESSEL_TANK:
                cv2.rectangle(
                    bev,
                    (pos_px[0] - 7, pos_px[1] - 7),
                    (pos_px[0] + 7, pos_px[1] + 7),
                    agent.primary_color_bgr,
                    -1
                )
                cv2.rectangle(
                    bev,
                    (pos_px[0] - 7, pos_px[1] - 7),
                    (pos_px[0] + 7, pos_px[1] + 7),
                    (0, 0, 255),
                    2
                )
            elif agent.agent_type == AgentType.FORKLIFT:
                cv2.rectangle(
                    bev,
                    (pos_px[0] - 9, pos_px[1] - 6),
                    (pos_px[0] + 9, pos_px[1] + 6),
                    agent.primary_color_bgr,
                    -1
                )
                cv2.circle(bev, pos_px, 3, (0, 0, 0), -1)

            # Heading direction vector

            hx = pos_px[0] + int(math.cos(agent.heading_rad) * 14)
            hy = pos_px[1] + int(math.sin(agent.heading_rad) * 14)
            cv2.arrowedLine(bev, pos_px, (hx, hy), (255, 255, 255), 1, tipLength=0.35)

            # Agent ID label
            cv2.putText(
                bev,
                f"{agent.id} ({agent.x:.1f}m, {agent.y:.1f}m)",
                (pos_px[0] + 9, pos_px[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Title OSD
        cv2.putText(
            bev,
            "FACTORY FLOOR DIGITAL TWIN - BIRD'S EYE VIEW (GROUND TRUTH)",
            (16, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 200),
            1,
            cv2.LINE_AA
        )
        return bev

    def stream_loop(self) -> Generator[Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, dict]], None, None]:
        """Generator yielding synchronized camera frames, BEV map, and ground truth."""
        self.is_running = True
        while self.is_running:
            start_time = time.time()
            frames, _, world_gt = self.step()
            bev_map = self.render_bird_eye_view()
            yield frames, bev_map, world_gt

            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.dt - elapsed)
            time.sleep(sleep_time)
