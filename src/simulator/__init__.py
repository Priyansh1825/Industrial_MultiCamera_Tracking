"""
Industrial Multi-Camera Tracking - Digital Simulation & Testing Environment.
Provides synthetic factory world modeling, virtual pinhole CCTV cameras,
ground truth generation, and benchmarking without physical camera hardware.
"""

from .factory_world import FactoryWorld, SimulatedAgent, AgentType, Zone, Obstacle
from .virtual_camera import VirtualCamera
from .stream_server import MultiCameraSimulatorServer
from .ground_truth_evaluator import MTMCTBenchmarkEvaluator

__all__ = [
    "FactoryWorld",
    "SimulatedAgent",
    "AgentType",
    "Zone",
    "Obstacle",
    "VirtualCamera",
    "MultiCameraSimulatorServer",
    "MTMCTBenchmarkEvaluator",
]
