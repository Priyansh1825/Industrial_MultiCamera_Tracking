# Industrial Multi-Camera Tracking & Spatial Analytics System (MTMCT)

Enterprise Multi-Target Multi-Camera Tracking (MTMCT) system for industrial factory floors, providing continuous asset and personnel tracking across complete CCTV coverage.

---

## Architecture Overview

```
[4x CCTV Camera Streams (1080p)]
                │
                ▼
  [Single-Camera Detection & Tracking]
  ├── YOLOv8 / YOLOv11 TensorRT Detector
  └── ByteTrack (Kalman Filter + 2-Stage Association)
                │
                ▼
  [Deep Re-Identification (ReID)]
  ├── Multi-Zone Circular Hue & Color Opponency Feature Extractor
  └── Vector Identity Gallery & Rolling Centroid Memory
                │
                ▼
  [Spatial Engine & Coordinate Homography]
  ├── Planar Homography Matrix H: 2D Footprints -> Metric Plant (X, Y)
  └── Spatio-Temporal Graph Optimizer (Hungarian Global Assignment)
                │
                ▼
  [Analytics, Storage & Live UI]
  ├── Safety Zone Intrusion & Dwell Time Analytics
  ├── FastAPI WebSockets Streaming Server (25 FPS)
  └── Responsive Digital Twin Canvas & CCTV Operator Dashboard
```

---

## Project Structure

```
Industrial_MultiCamera_Tracking/
├── configs/
│   └── system_config.yaml         # Camera extrinsics/intrinsics & network topology
├── docs/
│   ├── architecture_blueprint.md  # System design specifications
│   └── visual_project_plan.md     # Enterprise roadmap & phase analysis
├── src/
│   ├── api/
│   │   └── main.py                # FastAPI REST & WebSocket streaming endpoints
│   ├── dashboard/
│   │   └── index.html             # High-FPS Digital Twin live operator console
│   ├── detector/
│   │   └── yolo_detector.py       # YOLOv8/v11 TensorRT & PyTorch wrapper
│   ├── reid/
│   │   ├── feature_extractor.py   # Normalized embedding extractor (ONNX / Spatial-Color)
│   │   └── gallery.py             # Vector feature gallery & similarity search
│   ├── simulator/
│   │   ├── factory_world.py       # 2D/3D industrial plant physics simulation
│   │   ├── stream_server.py       # Synchronized multi-camera stream server
│   │   └── virtual_camera.py      # Pinhole 3D projective camera geometry
│   ├── spatial_engine/
│   │   ├── cross_camera_matcher.py# Spatio-Temporal Graph Optimizer & Global ID Fusion
│   │   ├── homography.py          # Planar homography image-to-world projector
│   │   └── spatial_analytics.py   # Factory hazard zones, speed checks & heatmaps
│   ├── tracker/
│   │   ├── byte_tracker.py        # Single-camera ByteTrack tracker & STrack lifecycle
│   │   ├── kalman_filter.py       # 8-state constant velocity Kalman filter
│   │   └── matching.py            # Bipartite matching & IoU cost matrices
│   └── mtmct_pipeline.py          # End-to-end synchronized orchestrator pipeline
├── tests/
│   ├── test_api.py                # REST API and query tests
│   ├── test_pipeline.py           # Multi-camera pipeline integration tests
│   ├── test_reid_spatial.py       # ReID and Spatio-Temporal Graph matcher tests
│   ├── test_simulator.py          # Digital twin camera projection tests
│   └── test_tracker.py            # ByteTracker and Kalman filter unit tests
├── requirements.txt
└── run_simulation.py              # CLI benchmark and testing runner
```

---

## Quickstart & Execution

### 1. Run Quantitative Benchmark
```powershell
python run_simulation.py --benchmark --frames 100
```

### 2. Run Test Suite & Linting
```powershell
python -m pytest tests/
python -m flake8 .
```

### 3. Launch Live Tracking API & Operator Dashboard
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://127.0.0.1:8000` in your web browser to view the real-time Digital Twin and CCTV surveillance feeds.
