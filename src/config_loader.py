"""
Configuration Loader for Industrial MTMCT System.
Loads and parses system_config.yaml with typed accessors.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


@dataclass
class CameraConfig:
    id: str
    name: str
    rtsp_url: Optional[str]
    fps: int
    resolution: Tuple[int, int]
    homography_matrix: Optional[List[List[float]]]
    adjacent_cameras: List[str]
    transition_time_bounds: Dict[str, Dict[str, float]]


@dataclass
class DetectionConfig:
    model_path: str
    confidence_threshold: float
    nms_iou_threshold: float
    classes: Dict[int, str]


@dataclass
class TrackingConfig:
    algorithm: str
    track_thresh: float
    track_buffer: int
    match_thresh: float


@dataclass
class ReIDConfig:
    person_model: str
    vessel_model: str
    embedding_dimension: int
    distance_metric: str
    similarity_threshold: float


@dataclass
class MTMCTEngineConfig:
    global_matching_window_sec: float
    spatial_consistency_weight: float
    appearance_consistency_weight: float
    max_velocity_mps: float


@dataclass
class DatabaseConfig:
    vector_db: Dict[str, Any]
    spatial_db: Dict[str, Any]
    message_broker: Dict[str, Any]


@dataclass
class SystemConfig:
    environment: str
    log_level: str
    inference_device: str
    tensorrt_enabled: bool
    fp16_precision: bool


@dataclass
class SimulationConfig:
    enabled: bool
    plant_dimensions_meters: Tuple[float, float]
    fps: int
    virtual_cameras: List[Dict[str, Any]]


class ConfigLoader:
    """Singleton config loader for system_config.yaml"""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            # Try multiple locations
            base = Path(__file__).parent.parent
            config_path = str(base / "configs" / "system_config.yaml")
            if not Path(config_path).exists():
                # Try from workspace root
                config_path = str(Path("D:/MY project/Industrial_MultiCamera_Tracking/configs/system_config.yaml"))

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)
        return self._config

    def get_config(self) -> Dict[str, Any]:
        if self._config is None:
            self.load()
        return self._config

    def get_system(self) -> SystemConfig:
        c = self.get_config()["system"]
        return SystemConfig(**c)

    def get_cameras(self) -> List[CameraConfig]:
        cams = []
        for cam in self.get_config()["camera_network"]["rtsp_streams"]:
            cams.append(CameraConfig(
                id=cam["id"],
                name=cam["name"],
                rtsp_url=cam.get("rtsp_url"),
                fps=cam["fps"],
                resolution=tuple(cam["resolution"]),
                homography_matrix=cam.get("homography_matrix"),
                adjacent_cameras=cam.get("adjacent_cameras", []),
                transition_time_bounds=cam.get("transition_time_bounds", {})
            ))
        return cams

    def get_camera_transitions(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Builds camera transition matrix from config."""
        transitions = {}
        for cam in self.get_config()["camera_network"]["rtsp_streams"]:
            cam_id = cam["id"]
            bounds = cam.get("transition_time_bounds", {})
            if bounds:
                transitions[cam_id] = {}
                for target_cam, tb in bounds.items():
                    transitions[cam_id][target_cam] = (tb["min_seconds"], tb["max_seconds"])
        return transitions

    def get_detection(self) -> DetectionConfig:
        c = self.get_config()["models"]["detection"]
        return DetectionConfig(**c)

    def get_tracking(self) -> TrackingConfig:
        c = self.get_config()["models"]["single_camera_tracking"]
        return TrackingConfig(**c)

    def get_reid(self) -> ReIDConfig:
        c = self.get_config()["models"]["reid_feature_extraction"]
        return ReIDConfig(**c)

    def get_mtmct_engine(self) -> MTMCTEngineConfig:
        c = self.get_config()["mtmct_engine"]
        return MTMCTEngineConfig(**c)

    def get_databases(self) -> DatabaseConfig:
        c = self.get_config()["databases"]
        return DatabaseConfig(**c)

    def get_simulation(self) -> SimulationConfig:
        c = self.get_config()["simulation"]
        return SimulationConfig(
            enabled=c["enabled"],
            plant_dimensions_meters=tuple(c["plant_dimensions_meters"]),
            fps=c["fps"],
            virtual_cameras=c["virtual_cameras"]
        )


config = ConfigLoader()