"""
Unit & Integration Tests for MTMCT FastAPI Endpoints.
"""
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mtmct-engine"


def test_zones_endpoint():
    response = client.get("/api/v1/zones")
    assert response.status_code == 200
    zones = response.json()
    assert len(zones) >= 3
    assert any(z["zone_id"] == "ZONE_HAZARD_REACTOR" for z in zones)


def test_search_and_trajectory_endpoint():
    # Execute a step to populate tracks
    from src.api.main import PIPELINE_ENGINE
    PIPELINE_ENGINE.step()

    # Query search
    response = client.get("/api/v1/search?query=VSL-402")
    assert response.status_code == 200

    # Query trajectory
    response_traj = client.get("/api/v1/trajectory?subject_id=VSL-402")
    assert response_traj.status_code == 200
    traj_data = response_traj.json()
    assert traj_data["subject_id"] == "VSL-402"
