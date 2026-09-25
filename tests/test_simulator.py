import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator.factory_world import AgentType, FactoryWorld, SimulatedAgent

from src.simulator.ground_truth_evaluator import MTMCTBenchmarkEvaluator
from src.simulator.stream_server import MultiCameraSimulatorServer
from src.simulator.virtual_camera import VirtualCamera


def test_factory_world_kinematics():
    """Verify agent movement, obstacle collision, and zone identification."""
    world = FactoryWorld(width_m=40.0, length_m=25.0)
    agent = SimulatedAgent(
        id="TEST-01",
        label="Test Worker",
        agent_type=AgentType.PERSON,
        x=5.0,
        y=5.0,
        target_speed_mps=1.5,
        waypoints=[(10.0, 5.0)]
    )
    world.add_agent(agent)

    initial_x = agent.x
    world.step(dt=1.0)

    # Agent should have moved towards the waypoint (dx > 0)
    assert agent.x > initial_x, "Agent did not move forward towards waypoint"

    # Verify ground truth dictionary
    gt = world.get_ground_truth()
    assert "TEST-01" in gt
    assert gt["TEST-01"]["class_name"] == "person"
    assert gt["TEST-01"]["world_x"] > initial_x


def test_virtual_camera_projection_and_homography():
    """Verify 3D to 2D projection and homography matrix inversion accuracy."""
    cam = VirtualCamera(
        camera_id="CAM_TEST",
        name="Test Camera",
        pos_3d=(2.0, 2.0, 5.0),
        target_3d=(10.0, 10.0, 0.0),
        resolution=(1280, 720),
        fov_deg=75.0
    )

    world = FactoryWorld(width_m=40.0, length_m=25.0)
    agent = SimulatedAgent(
        id="EMP-100",
        label="Test Person",
        agent_type=AgentType.PERSON,
        x=8.0,
        y=8.0,
        height_m=1.75
    )
    world.add_agent(agent)

    # Project agent to camera
    proj = cam.get_agent_bounding_box(agent)
    assert proj is not None, "Agent at (8, 8) should be visible in camera field of view"
    x1, y1, x2, y2, fu, fv = proj
    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720

    # Test Homography Inverse Mapping: Image (fu, fv) -> Ground (X, Y)
    p_img = np.array([fu, fv, 1.0], dtype=np.float64)
    p_world = cam.H_img_to_world @ p_img
    recovered_x = p_world[0] / p_world[2]
    recovered_y = p_world[1] / p_world[2]

    # Error should be less than 5 centimeters
    assert abs(recovered_x - 8.0) < 0.05, f"Homography recovery error X: {recovered_x}"
    assert abs(recovered_y - 8.0) < 0.05, f"Homography recovery error Y: {recovered_y}"


def test_multi_camera_stream_server():
    """Verify synchronized 4-camera frame generation and BEV map rendering."""
    server = MultiCameraSimulatorServer(fps=25)
    frames, cam_dets, world_gt = server.step()

    assert len(frames) == 4, "Should generate 4 camera streams"
    for cid, f in frames.items():
        assert f.shape == (720, 1280, 3), f"Stream {cid} has wrong frame dimensions"

    bev_map = server.render_bird_eye_view(800, 500)
    assert bev_map.shape == (500, 800, 3), "BEV map has wrong dimensions"


def test_benchmark_evaluator_metrics():
    """Verify MOTA, precision, recall, and localization error calculations."""
    evaluator = MTMCTBenchmarkEvaluator()

    gt = [{"global_id": "EMP-01", "bbox": (100.0, 100.0, 200.0, 300.0), "world_coords": (5.0, 10.0)}]
    pred = [{"track_id": 101, "bbox": (102.0, 98.0, 201.0, 302.0), "world_coords": (5.05, 10.02)}]

    res = evaluator.evaluate_camera_frame(0, "CAM_01", gt, pred)
    assert res.true_positives == 1
    assert res.false_positives == 0
    assert res.false_negatives == 0
    assert res.id_switches == 0

    metrics = evaluator.get_summary_metrics()
    assert metrics["mota"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mean_localization_error_m"] < 0.1


if __name__ == "__main__":
    test_factory_world_kinematics()
    test_virtual_camera_projection_and_homography()
    test_multi_camera_stream_server()
    test_benchmark_evaluator_metrics()
    print("[Tests] All digital simulator tests passed successfully!")
