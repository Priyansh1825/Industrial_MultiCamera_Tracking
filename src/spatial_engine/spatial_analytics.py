"""
Spatial Factory Analytics: Dwell Time Monitoring, Safety Zone Intrusion, and Heatmap Generation.
"""
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FactoryZone:
    zone_id: str
    name: str
    polygon: List[Tuple[float, float]]  # List of (X, Y) vertices in plant meters
    is_hazard: bool = False
    max_authorized_speed_mps: float = 2.0
    allowed_classes: List[str] = field(default_factory=lambda: ["person", "vessel_tank", "forklift"])


@dataclass
class SafetyAlert:
    alert_id: str
    timestamp: float
    zone_id: str
    zone_name: str
    subject_id: str
    subject_label: str
    alert_type: str  # "UNAUTHORIZED_INTRUSION", "SPEED_VIOLATION", "OVERSTAY"
    severity: str    # "CRITICAL", "WARNING", "INFO"
    message: str


class SpatialAnalyticsEngine:
    """
    Monitors factory zones, calculates personnel dwell times, and detects safety violations in real time.
    """
    def __init__(self, plant_dimensions: Tuple[float, float] = (40.0, 25.0)):
        self.plant_width, self.plant_length = plant_dimensions
        self.zones: Dict[str, FactoryZone] = {}
        self.subject_zone_occupancy: Dict[str, Dict[str, float]] = {}  # {subject_id: {zone_id: enter_timestamp}}
        self.total_zone_dwell_times: Dict[str, Dict[str, float]] = {}  # {subject_id: {zone_id: total_seconds}}
        self.active_alerts: List[SafetyAlert] = []
        self.alert_counter = 1

        self._initialize_default_factory_zones()

    def _initialize_default_factory_zones(self):
        """Sets up standard industrial factory floor zones."""
        # 1. Chemical Reaction Bay (Hazardous Area)
        self.add_zone(FactoryZone(
            zone_id="ZONE_HAZARD_REACTOR",
            name="Hazardous Chemical Reaction Bay",
            polygon=[(20.0, 0.0), (32.0, 0.0), (32.0, 10.0), (20.0, 10.0)],
            is_hazard=True,
            allowed_classes=["person", "vessel_tank"]
        ))

        # 2. Automated Forklift Transit Corridor
        self.add_zone(FactoryZone(
            zone_id="ZONE_FORKLIFT_LANE",
            name="High-Speed Forklift Corridor",
            polygon=[(0.0, 10.0), (40.0, 10.0), (40.0, 14.0), (0.0, 14.0)],
            is_hazard=False,
            max_authorized_speed_mps=3.5,
            allowed_classes=["forklift", "pallet_jack"]
        ))

        # 3. Main Logistics Loading Dock
        self.add_zone(FactoryZone(
            zone_id="ZONE_LOADING_DOCK",
            name="South Logistics Loading Dock",
            polygon=[(28.0, 14.0), (40.0, 14.0), (40.0, 25.0), (28.0, 25.0)],
            is_hazard=False,
            allowed_classes=["person", "forklift", "vessel_tank"]
        ))

    def add_zone(self, zone: FactoryZone):
        self.zones[zone.zone_id] = zone

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Ray-casting algorithm to test if (x, y) coordinate is within polygon boundaries."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def update_subject_telemetry(
        self,
        subject_id: str,
        label: str,
        class_name: str,
        world_coords: Tuple[float, float],
        timestamp: Optional[float] = None,
        speed_mps: Optional[float] = None
    ) -> List[SafetyAlert]:
        """
        Processes real-time subject coordinate update, updates dwell times, and triggers safety alerts.
        """
        now = timestamp or time.time()
        new_alerts: List[SafetyAlert] = []

        if subject_id not in self.subject_zone_occupancy:
            self.subject_zone_occupancy[subject_id] = {}
            self.total_zone_dwell_times[subject_id] = {}

        current_inside_zones = set()

        for zone_id, zone in self.zones.items():
            is_inside = self.point_in_polygon(world_coords, zone.polygon)

            if is_inside:
                current_inside_zones.add(zone_id)

                # 1. Track entry timestamp & dwell time
                if zone_id not in self.subject_zone_occupancy[subject_id]:
                    self.subject_zone_occupancy[subject_id][zone_id] = now
                else:
                    elapsed = now - self.subject_zone_occupancy[subject_id][zone_id]
                    self.total_zone_dwell_times[subject_id][zone_id] = (
                        self.total_zone_dwell_times[subject_id].get(zone_id, 0.0) + elapsed
                    )
                    self.subject_zone_occupancy[subject_id][zone_id] = now

                # 2. Check for hazard or unauthorized access
                if zone.is_hazard and class_name not in zone.allowed_classes:
                    msg = f"{label} ({class_name}) entered restricted hazard zone '{zone.name}'"
                    alert = SafetyAlert(
                        alert_id=f"ALT-{self.alert_counter:04d}",
                        timestamp=now,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        subject_id=subject_id,
                        subject_label=label,
                        alert_type="UNAUTHORIZED_INTRUSION",
                        severity="CRITICAL",
                        message=msg
                    )
                    self.alert_counter += 1
                    self.active_alerts.append(alert)
                    new_alerts.append(alert)

                # 3. Check speed violation in regulated zones
                if speed_mps is not None and speed_mps > zone.max_authorized_speed_mps:
                    msg = (
                        f"{label} speed ({speed_mps:.1f} m/s) exceeded "
                        f"limit ({zone.max_authorized_speed_mps:.1f} m/s)"
                    )
                    alert = SafetyAlert(
                        alert_id=f"ALT-{self.alert_counter:04d}",
                        timestamp=now,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        subject_id=subject_id,
                        subject_label=label,
                        alert_type="SPEED_VIOLATION",
                        severity="WARNING",
                        message=msg
                    )
                    self.alert_counter += 1
                    self.active_alerts.append(alert)
                    new_alerts.append(alert)

            else:
                # Subject left the zone
                if zone_id in self.subject_zone_occupancy[subject_id]:
                    del self.subject_zone_occupancy[subject_id][zone_id]

        return new_alerts

    def generate_heatmap_raster(
        self,
        trajectory_points: List[Tuple[float, float]],
        resolution: Tuple[int, int] = (250, 400)
    ) -> np.ndarray:
        """
        Generates a 2D floor occupancy density heatmap matrix from historical spatial coordinates.
        resolution: (height, width)
        """
        h, w = resolution
        heatmap = np.zeros((h, w), dtype=np.float32)

        for x, y in trajectory_points:
            px = int(np.clip((x / self.plant_width) * w, 0, w - 1))
            py = int(np.clip((y / self.plant_length) * h, 0, h - 1))
            heatmap[py, px] += 1.0

        import cv2
        # Apply Gaussian blur for smooth density estimation
        blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
        max_v = np.max(blurred)
        if max_v > 1e-6:
            normalized = (blurred / max_v * 255).astype(np.uint8)
        else:
            normalized = blurred.astype(np.uint8)

        color_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return color_heatmap
