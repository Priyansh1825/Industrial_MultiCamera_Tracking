"""
Industrial Multi-Camera Tracking - Digital Simulation & Testing Runner.
Runs synchronized multi-camera CCTV feeds, bird's-eye digital twin map,
and automated ground-truth benchmarks on zero hardware budget.
"""

import argparse
import time
from typing import Dict

import cv2
import numpy as np

from src.simulator.factory_world import AgentType, SimulatedAgent
from src.simulator.ground_truth_evaluator import MTMCTBenchmarkEvaluator
from src.simulator.stream_server import MultiCameraSimulatorServer


def build_unified_dashboard_frame(
    cam_frames: Dict[str, np.ndarray],
    bev_map: np.ndarray,
    fps_val: float,
    active_count: int,
    paused: bool = False
) -> np.ndarray:
    """
    Composes a 1920x1080 high-resolution unified operator dashboard
    containing a 2x2 CCTV camera grid on the left and the 2D Digital Twin on the right.
    """
    canvas = np.full((1080, 1920, 3), (20, 22, 26), dtype=np.uint8)

    # 1. Resize and arrange 4 CCTV feeds in 2x2 grid on left side (width: 1080, height: 1080)
    cams = list(cam_frames.values())
    grid_w, grid_h = 530, 300

    cam_thumbnails = []
    for f in cams[:4]:
        thumb = cv2.resize(f, (grid_w, grid_h))
        cam_thumbnails.append(thumb)

    while len(cam_thumbnails) < 4:
        blank = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        cv2.putText(blank, "OFFLINE", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
        cam_thumbnails.append(blank)

    top_row = np.hstack([cam_thumbnails[0], cam_thumbnails[1]])
    bot_row = np.hstack([cam_thumbnails[2], cam_thumbnails[3]])
    cctv_grid = np.vstack([top_row, bot_row])

    # Place CCTV grid at (20, 80)
    canvas[80:80 + 600, 20:20 + 1060] = cctv_grid

    # 2. Resize and place 2D Digital Twin Floor Map on right side
    bev_resized = cv2.resize(bev_map, (800, 600))
    canvas[80:80 + 600, 1100:1100 + 800] = bev_resized

    # 3. Bottom Information & Metric Telemetry Panel
    cv2.rectangle(canvas, (20, 700), (1900, 1050), (28, 32, 38), -1)
    cv2.rectangle(canvas, (20, 700), (1900, 1050), (60, 65, 75), 1)

    cv2.putText(
        canvas,
        "REAL-TIME SIMULATION & TRACKING TELEMETRY",
        (40, 735),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 200),
        2,
        cv2.LINE_AA
    )

    # Telemetry data columns
    status_str = "PAUSED" if paused else "STREAMING (LIVE)"
    status_color = (0, 165, 255) if paused else (0, 255, 0)

    cv2.putText(canvas, f"Engine Status: {status_str}", (40, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 1)
    cv2.putText(canvas, f"Simulated FPS: {fps_val:.1f} Hz", (40, 815),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
    cv2.putText(canvas, f"Active Tracked Subjects: {active_count}", (40, 850),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
    cv2.putText(canvas, "Virtual Cameras: 4 Active Streams (1080p)", (40, 885),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    # Keybind shortcuts guide
    cv2.putText(canvas, "INTERACTIVE CONTROLS:", (450, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)
    cv2.putText(canvas, "[SPACE] : Pause / Resume Simulation", (450, 815),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "[A]     : Spawn Additional Worker / Vessel", (450, 850),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "[Q/ESC] : Exit Digital Testing Environment", (450, 885),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # System specs note
    cv2.putText(canvas, "DIGITAL TESTING ENVIRONMENT SPECIFICATIONS:", (920, 780),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 100), 1)
    cv2.putText(canvas, "Plant Dimensions: 40.0m x 25.0m Concrete Epoxy Floor", (920, 815),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "Floor Projection: Planar Homography H & 3D Pinhole Raycasting", (920, 850),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "Zero-Hardware Validation: Full Ground-Truth vs Predicted Matching", (920, 885),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Top Header Banner
    cv2.rectangle(canvas, (0, 0), (1920, 60), (14, 16, 20), -1)
    cv2.putText(
        canvas,
        "INDUSTRIAL MULTI-CAMERA TRACKING (MTMCT) - DIGITAL TESTING SUITE & DIGITAL TWIN",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "STATUS: SYNTHETIC GROUND TRUTH BROADCASTING",
        (1400, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 120),
        1,
        cv2.LINE_AA
    )

    return canvas


def run_benchmark_mode(num_frames: int = 150) -> None:
    """Executes headless simulation and prints quantitative tracking metrics."""
    print(f"\n[Benchmark] Initializing Digital Testing Suite for {num_frames} frames...")
    server = MultiCameraSimulatorServer(fps=30)
    evaluator = MTMCTBenchmarkEvaluator()

    start_t = time.time()
    for f_idx in range(num_frames):
        frames, cam_detections, _ = server.step()

        # Simulate detector/tracker with 96% detection recall and small homography noise
        for cam_id, gt_dets in cam_detections.items():
            sim_preds = []
            for d in gt_dets:
                # 96% detection rate
                if np.random.rand() < 0.96:
                    x1, y1, x2, y2 = d.bbox
                    # Slight 2-pixel jitter
                    x1 += np.random.uniform(-2, 2)
                    y1 += np.random.uniform(-2, 2)
                    x2 += np.random.uniform(-2, 2)
                    y2 += np.random.uniform(-2, 2)

                    wx, wy = d.world_coords
                    # 5cm homography projection noise
                    pred_wx = wx + np.random.normal(0, 0.05)
                    pred_wy = wy + np.random.normal(0, 0.05)

                    # Hash ID for tracking consistency
                    track_id = abs(hash(d.global_id)) % 10000

                    sim_preds.append({
                        "track_id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "world_coords": (pred_wx, pred_wy)
                    })

            gt_dicts = [
                {
                    "global_id": d.global_id,
                    "bbox": d.bbox,
                    "world_coords": d.world_coords
                }
                for d in gt_dets
            ]
            evaluator.evaluate_camera_frame(f_idx, cam_id, gt_dicts, sim_preds)

    elapsed = time.time() - start_t
    fps = num_frames / elapsed

    print(f"[Benchmark] Completed {num_frames} multi-camera frames in {elapsed:.2f}s ({fps:.1f} FPS).")
    print(evaluator.generate_report_markdown())


def run_gui_mode() -> None:
    """Launches the interactive real-time multi-camera dashboard window."""
    print("\n[GUI] Launching Industrial MTMCT Digital Testing Environment...")
    print("[GUI] Press 'Q' or ESC in the window to exit, SPACE to pause/resume, 'A' to spawn agent.")

    server = MultiCameraSimulatorServer(fps=25)
    paused = False

    agent_counter = 1
    fps_history = []

    window_name = "Industrial MTMCT - Digital Twin & Camera Simulator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    while True:
        t0 = time.time()
        if not paused:
            frames, _, world_gt = server.step()
        else:
            frames = {cid: cam.render_frame(server.world, True) for cid, cam in server.cameras.items()}
            world_gt = server.world.get_ground_truth()

        bev_map = server.render_bird_eye_view()

        t1 = time.time()
        curr_fps = 1.0 / max(1e-5, (t1 - t0))
        fps_history.append(curr_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        dashboard = build_unified_dashboard_frame(
            cam_frames=frames,
            bev_map=bev_map,
            fps_val=avg_fps,
            active_count=len(world_gt),
            paused=paused
        )

        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # ESC or Q
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif key in (ord('a'), ord('A')):
            # Spawn new dynamic worker
            new_id = f"EMP-9{agent_counter:02d}"
            agent_counter += 1
            new_agent = SimulatedAgent(
                id=new_id,
                label=f"Extra Field Operator #{agent_counter}",
                agent_type=AgentType.PERSON,
                x=np.random.uniform(3.0, 35.0),
                y=np.random.uniform(3.0, 22.0),
                target_speed_mps=np.random.uniform(1.1, 1.5),
                primary_color_bgr=(np.random.randint(0, 255), np.random.randint(150, 255), np.random.randint(0, 255))
            )
            server.world.add_agent(new_agent)
            print(f"[GUI] Spawned new agent: {new_id} at ({new_agent.x:.1f}m, {new_agent.y:.1f}m)")

    cv2.destroyAllWindows()
    print("[GUI] Digital Testing Environment terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Industrial MTMCT Digital Testing Environment")
    parser.add_argument("--benchmark", action="store_true", help="Run quantitative accuracy benchmark mode")
    parser.add_argument("--frames", type=int, default=150, help="Number of benchmark frames to simulate")
    parser.add_argument("--gui", action="store_true", help="Launch interactive multi-camera dashboard window")

    args = parser.parse_args()

    if args.benchmark or not args.gui:
        run_benchmark_mode(num_frames=args.frames)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
