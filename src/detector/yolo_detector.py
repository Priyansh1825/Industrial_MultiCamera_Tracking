"""
YOLOv8 / YOLOv11 TensorRT & PyTorch Detection Wrapper for Industrial Assets and Personnel.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    world_coords: Optional[Tuple[float, float]] = None  # (X, Y) in plant meters


class FactoryObjectDetector:
    def __init__(self, model_path: str = "yolov8x.pt", conf_threshold: float = 0.45, iou_threshold: float = 0.50):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = {
            0: "person",
            1: "vessel_tank",
            2: "forklift",
            3: "pallet_jack"
        }
        self.model = None

    def load_model(self):
        """Loads YOLO PyTorch or TensorRT compiled engine."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"[Detector] Loaded model from {self.model_path}")
        except Exception as e:
            print(f"[Detector] Warning: Could not initialize model engine: {e}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs inference on a video frame and returns structured Detection objects."""
        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))
        return detections
