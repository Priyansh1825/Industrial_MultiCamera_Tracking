"""
Factory Floor World Simulator: Models industrial plant physics, zones, obstacles,
and kinematic movement of workers, vessels, and material handling equipment.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import random
import time
from typing import Dict, List, Optional, Tuple


class AgentType(Enum):
    PERSON = 0
    VESSEL_TANK = 1
    FORKLIFT = 2
    PALLET_JACK = 3


@dataclass
class Zone:
    id: str
    name: str
    x_min: float  # In meters
    y_min: float
    x_max: float
    y_max: float
    color_rgb: Tuple[int, int, int] = (60, 60, 60)
    zone_type: str = "operational"

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass
class Obstacle:
    id: str
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    height_m: float = 2.5
    color_rgb: Tuple[int, int, int] = (100, 110, 120)

    def collides_with(self, x: float, y: float, margin: float = 0.5) -> bool:
        return (
            (self.x_min - margin <= x <= self.x_max + margin)
            and (self.y_min - margin <= y <= self.y_max + margin)
        )


@dataclass
class SimulatedAgent:
    id: str

    label: str
    agent_type: AgentType
    x: float
    y: float
    target_speed_mps: float = 1.2
    heading_rad: float = 0.0
    width_m: float = 0.6
    length_m: float = 0.6
    height_m: float = 1.75
    primary_color_bgr: Tuple[int, int, int] = (0, 215, 255)  # High-vis yellow
    secondary_color_bgr: Tuple[int, int, int] = (40, 40, 40)
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    current_waypoint_idx: int = 0
    dwell_timer_sec: float = 0.0
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # [(x, y, timestamp)]
    max_history: int = 200

    def update(self, dt: float, world: "FactoryWorld") -> None:
        """Updates agent position, waypoint navigation, and collision avoidance."""
        now = time.time()
        self.history.append((self.x, self.y, now))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Handle dwelling at a waypoint
        if self.dwell_timer_sec > 0:
            self.dwell_timer_sec -= dt
            return

        if not self.waypoints:
            self._generate_random_waypoints(world)
            return

        # Navigate to current waypoint
        tx, ty = self.waypoints[self.current_waypoint_idx]
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)

        if dist < 0.8:
            # Reached waypoint -> advance or dwell
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            if random.random() < 0.35:
                self.dwell_timer_sec = random.uniform(2.0, 8.0)
            return

        # Compute desired heading
        target_heading = math.atan2(dy, dx)
        heading_diff = (target_heading - self.heading_rad + math.pi) % (2 * math.pi) - math.pi
        self.heading_rad += max(-3.0 * dt, min(3.0 * dt, heading_diff))

        # Velocity step
        vx = math.cos(self.heading_rad) * self.target_speed_mps
        vy = math.sin(self.heading_rad) * self.target_speed_mps

        next_x = self.x + vx * dt
        next_y = self.y + vy * dt

        # Boundary checks
        margin = 0.8
        next_x = max(margin, min(world.width_m - margin, next_x))
        next_y = max(margin, min(world.length_m - margin, next_y))

        # Obstacle avoidance
        collided = False
        for obs in world.obstacles:
            if obs.collides_with(next_x, next_y, margin=self.width_m / 2.0 + 0.2):
                collided = True
                break

        if collided:
            # Shift heading and find alternative path
            self.heading_rad += random.choice([-1.0, 1.0]) * (math.pi / 3.0)
            self._generate_random_waypoints(world)
        else:
            self.x = next_x
            self.y = next_y

    def _generate_random_waypoints(self, world: "FactoryWorld") -> None:
        """Selects a sequence of reachable zones as waypoints."""
        available_zones = list(world.zones.values())
        if not available_zones:
            self.waypoints = [
                (random.uniform(2.0, world.width_m - 2.0),
                 random.uniform(2.0, world.length_m - 2.0))
                for _ in range(4)
            ]
        else:
            selected_zones = random.sample(available_zones, min(4, len(available_zones)))
            self.waypoints = [z.center for z in selected_zones]
        self.current_waypoint_idx = 0


class FactoryWorld:
    """Simulated 2D/3D Industrial Plant Environment."""

    def __init__(self, width_m: float = 40.0, length_m: float = 25.0):
        self.width_m = width_m
        self.length_m = length_m
        self.zones: Dict[str, Zone] = {}
        self.obstacles: List[Obstacle] = []
        self.agents: Dict[str, SimulatedAgent] = {}
        self.current_time_sec: float = 0.0

        self._setup_default_industrial_layout()

    def _setup_default_industrial_layout(self) -> None:
        """Initializes standard factory floor zones, aisles, and machinery."""
        # 1. Factory Zones
        self.zones = {
            "ZONE_NORTH_GATE": Zone(
                id="ZONE_NORTH_GATE",
                name="Main North Entrance Gate",
                x_min=1.0, y_min=1.0, x_max=12.0, y_max=8.0,
                color_rgb=(40, 80, 50),
                zone_type="entry_exit"
            ),
            "ZONE_AISLE_WEST": Zone(
                id="ZONE_AISLE_WEST",
                name="Aisle West Processing",
                x_min=1.0, y_min=9.0, x_max=14.0, y_max=24.0,
                color_rgb=(50, 60, 90),
                zone_type="corridor"
            ),
            "ZONE_REACTOR_BAY": Zone(
                id="ZONE_REACTOR_BAY",
                name="Chemical Reactor & Agitation Bay",
                x_min=15.0, y_min=1.0, x_max=27.0, y_max=14.0,
                color_rgb=(100, 60, 30),
                zone_type="hazardous_processing"
            ),
            "ZONE_ASSEMBLY_CENTRAL": Zone(
                id="ZONE_ASSEMBLY_CENTRAL",
                name="Central Assembly & Packaging",
                x_min=15.0, y_min=15.0, x_max=27.0, y_max=24.0,
                color_rgb=(70, 70, 40),
                zone_type="assembly"
            ),
            "ZONE_LOADING_DOCK": Zone(
                id="ZONE_LOADING_DOCK",
                name="South Loading & Dispatch Dock",
                x_min=28.0, y_min=1.0, x_max=39.0, y_max=24.0,
                color_rgb=(60, 40, 70),
                zone_type="logistics"
            ),
        }

        # 2. Obstacles & Machinery
        self.obstacles = [
            Obstacle("MACH_01", "CNC Milling Station #1", 4.0, 11.0, 7.5, 15.0, 2.8),
            Obstacle("MACH_02", "Automated Packaging Line", 4.0, 17.5, 7.5, 21.5, 2.4),
            Obstacle("TANK_BANK_A", "Raw Material Chemical Silos", 18.0, 3.0, 24.0, 7.0, 4.5),
            Obstacle("RACK_STORAGE", "High-Bay Heavy Pallet Racks", 31.0, 5.0, 36.0, 19.0, 5.0),
            Obstacle("PILLAR_01", "Support Pillar C1", 14.5, 7.0, 15.5, 8.0, 6.0),
            Obstacle("PILLAR_02", "Support Pillar C2", 14.5, 17.0, 15.5, 18.0, 6.0),
            Obstacle("PILLAR_03", "Support Pillar C3", 27.5, 7.0, 28.5, 8.0, 6.0),
            Obstacle("PILLAR_04", "Support Pillar C4", 27.5, 17.0, 28.5, 18.0, 6.0),
        ]

    def add_agent(self, agent: SimulatedAgent) -> None:
        """Registers a dynamic agent in the simulation."""
        self.agents[agent.id] = agent

    def spawn_default_fleet(self) -> None:
        """Spawns realistic industrial workforce and equipment fleet."""
        # 1. Operators / Personnel (Uniforms & PPE)
        workers = [
            SimulatedAgent(
                id="EMP-108",
                label="Operator J. Smith (North Shift)",
                agent_type=AgentType.PERSON,
                x=3.5, y=4.2,
                target_speed_mps=1.35,
                primary_color_bgr=(0, 215, 255),  # High-vis neon yellow PPE
                secondary_color_bgr=(30, 30, 30),
                waypoints=[(3.5, 4.2), (3.5, 16.0), (16.0, 18.0), (10.0, 5.0)]
            ),
            SimulatedAgent(
                id="EMP-204",
                label="Technician M. Davis (Reactor Spec)",
                agent_type=AgentType.PERSON,
                x=19.5, y=9.5,
                target_speed_mps=1.2,
                primary_color_bgr=(0, 140, 255),  # High-vis bright orange PPE
                secondary_color_bgr=(20, 20, 20),
                waypoints=[(19.5, 9.5), (25.0, 4.0), (16.0, 4.0), (22.0, 11.0)]
            ),
            SimulatedAgent(
                id="EMP-312",
                label="Logistics Supervisor R. Patel",
                agent_type=AgentType.PERSON,
                x=30.0, y=12.0,
                target_speed_mps=1.4,
                primary_color_bgr=(50, 205, 50),  # Lime green safety vest
                secondary_color_bgr=(40, 40, 40),
                waypoints=[(30.0, 3.0), (30.0, 21.0), (37.0, 21.0), (37.0, 3.0)]
            )
        ]

        # 2. Chemical Reactor Vessels (Mobile Tanks on Wheels)
        vessels = [
            SimulatedAgent(
                id="VSL-402",
                label="Chemical Reactor Vessel #402",
                agent_type=AgentType.VESSEL_TANK,
                x=21.0, y=10.0,
                target_speed_mps=0.8,
                width_m=1.2, length_m=1.2, height_m=1.6,
                primary_color_bgr=(180, 180, 180),  # Stainless steel / silver
                secondary_color_bgr=(200, 50, 50),  # Blue HAZMAT band
                waypoints=[(21.0, 10.0), (16.0, 10.0), (10.0, 12.0), (3.0, 6.0)]
            ),
            SimulatedAgent(
                id="VSL-501",
                label="Agitation Buffer Tank #501",
                agent_type=AgentType.VESSEL_TANK,
                x=17.0, y=5.0,
                target_speed_mps=0.7,
                width_m=1.1, length_m=1.1, height_m=1.5,
                primary_color_bgr=(160, 160, 190),
                secondary_color_bgr=(50, 150, 50),
                waypoints=[(17.0, 5.0), (24.0, 12.0), (29.0, 8.0), (20.0, 4.0)]
            )
        ]

        # 3. Forklifts & Pallet Jacks
        vehicles = [
            SimulatedAgent(
                id="FLT-01",
                label="Forklift Unit #01 (Loading Bay)",
                agent_type=AgentType.FORKLIFT,
                x=29.0, y=4.0,
                target_speed_mps=2.4,
                width_m=1.4, length_m=2.2, height_m=2.1,
                primary_color_bgr=(0, 200, 255),  # Safety yellow
                secondary_color_bgr=(10, 10, 10),
                waypoints=[(29.0, 3.0), (29.0, 22.0), (20.0, 20.0), (29.0, 10.0)]
            )
        ]

        for a in workers + vessels + vehicles:
            self.add_agent(a)

    def step(self, dt: float) -> None:
        """Advances physics and kinematic agents by dt seconds."""
        self.current_time_sec += dt
        for agent in self.agents.values():
            agent.update(dt, self)

    def get_zone_for_position(self, x: float, y: float) -> Optional[Zone]:
        """Finds which named factory zone contains point (x, y)."""
        for zone in self.zones.values():
            if zone.contains(x, y):
                return zone
        return None

    def get_ground_truth(self) -> Dict[str, dict]:
        """Returns structured ground truth state of all agents."""
        gt = {}
        for aid, a in self.agents.items():
            zone = self.get_zone_for_position(a.x, a.y)
            gt[aid] = {
                "id": a.id,
                "label": a.label,
                "class_id": a.agent_type.value,
                "class_name": a.agent_type.name.lower(),
                "world_x": round(a.x, 3),
                "world_y": round(a.y, 3),
                "speed_mps": round(a.target_speed_mps, 2),
                "heading_deg": round(math.degrees(a.heading_rad), 1),
                "zone_id": zone.id if zone else "UNASSIGNED",
                "zone_name": zone.name if zone else "Corridor",
                "timestamp": round(time.time(), 3),
            }
        return gt
