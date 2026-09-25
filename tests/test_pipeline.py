"""
Integration Test for End-to-End MTMCT Pipeline Engine.
"""
from src.mtmct_pipeline import MTMCTPipelineEngine


def test_mtmct_pipeline_step_execution():
    engine = MTMCTPipelineEngine(fps=25)

    # Run 5 consecutive synchronized pipeline steps
    for _ in range(5):
        frames, global_tracks, alerts = engine.step()

        # Must produce frames for all 4 virtual CCTV cameras
        assert len(frames) == 4
        for cam_id, frame in frames.items():
            assert frame.shape == (720, 1280, 3)

        # Global tracks should be populated
        assert len(global_tracks) > 0
        for gid, gt in global_tracks.items():
            assert gt.global_id != ""
            assert len(gt.trajectory) > 0
            assert 0.0 <= gt.world_coords[0] <= 40.0
            assert 0.0 <= gt.world_coords[1] <= 25.0
