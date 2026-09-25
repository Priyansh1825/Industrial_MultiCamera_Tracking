"""
MTMCT Benchmark Evaluator:
Calculates standard Multi-Target Multi-Camera Tracking metrics (MOTA, IDF1, ID Switches,
Localization Error in meters) by comparing tracker outputs against simulator ground truth.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple


@dataclass
class FrameEvaluationResult:
    frame_idx: int
    camera_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    localization_errors_m: List[float] = field(default_factory=list)


class MTMCTBenchmarkEvaluator:
    """Computes comprehensive Multi-Target Multi-Camera Tracking accuracy metrics."""

    def __init__(self, iou_threshold: float = 0.5, spatial_distance_thresh_m: float = 2.0):
        self.iou_threshold = iou_threshold
        self.spatial_distance_thresh_m = spatial_distance_thresh_m

        # Global counters
        self.total_gt_objects = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.localization_errors_m: List[float] = []

        # Track history mapping: gt_id -> last assigned pred_track_id
        self.gt_to_pred_map: Dict[str, int] = {}
        self.frame_history: List[FrameEvaluationResult] = []

    def evaluate_camera_frame(
        self,
        frame_idx: int,
        camera_id: str,
        ground_truth: List[dict],  # [{global_id, bbox, world_coords}]
        predictions: List[dict]    # [{track_id, bbox, world_coords}]
    ) -> FrameEvaluationResult:
        """
        Evaluates predictions in a single camera frame against ground truth.
        """
        num_gt = len(ground_truth)
        self.total_gt_objects += num_gt

        if num_gt == 0:
            fp = len(predictions)
            self.total_fp += fp
            res = FrameEvaluationResult(frame_idx, camera_id, 0, fp, 0, 0, [])
            self.frame_history.append(res)
            return res

        matched_gt = set()
        matched_pred = set()
        frame_errors: List[float] = []
        frame_idsw = 0

        # Match bounding boxes via IoU
        for g_idx, gt in enumerate(ground_truth):
            best_iou = 0.0
            best_p_idx = -1

            for p_idx, pred in enumerate(predictions):
                if p_idx in matched_pred:
                    continue
                iou = self._compute_iou(gt["bbox"], pred["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx

            if best_iou >= self.iou_threshold and best_p_idx != -1:
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)

                pred_match = predictions[best_p_idx]
                gt_id = gt["global_id"]
                pred_id = pred_match["track_id"]

                # Check ID Switch
                if gt_id in self.gt_to_pred_map and self.gt_to_pred_map[gt_id] != pred_id:
                    frame_idsw += 1
                    self.total_id_switches += 1
                self.gt_to_pred_map[gt_id] = pred_id

                # Calculate Metric Localization Error (meters)
                if "world_coords" in gt and "world_coords" in pred_match and pred_match["world_coords"] is not None:
                    gt_x, gt_y = gt["world_coords"]
                    pr_x, pr_y = pred_match["world_coords"]
                    err = math.hypot(gt_x - pr_x, gt_y - pr_y)
                    frame_errors.append(err)
                    self.localization_errors_m.append(err)

        tp = len(matched_gt)
        fn = num_gt - tp
        fp = len(predictions) - len(matched_pred)

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        res = FrameEvaluationResult(
            frame_idx=frame_idx,
            camera_id=camera_id,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            id_switches=frame_idsw,
            localization_errors_m=frame_errors
        )
        self.frame_history.append(res)
        return res

    def get_summary_metrics(self) -> Dict[str, float]:
        """Calculates final MOTA, Precision, Recall, F1, and mean localization error."""
        precision = self.total_tp / max(1, self.total_tp + self.total_fp)
        recall = self.total_tp / max(1, self.total_tp + self.total_fn)
        f1 = (2 * precision * recall) / max(1e-6, precision + recall)

        mota = 1.0 - (self.total_fn + self.total_fp + self.total_id_switches) / max(1, self.total_gt_objects)
        mota = max(0.0, mota)

        mean_loc_err = (sum(self.localization_errors_m) / len(self.localization_errors_m)
                        if self.localization_errors_m else 0.0)

        return {
            "total_ground_truth": float(self.total_gt_objects),
            "true_positives": float(self.total_tp),
            "false_positives": float(self.total_fp),
            "false_negatives": float(self.total_fn),
            "id_switches": float(self.total_id_switches),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mota": round(mota, 4),
            "mean_localization_error_m": round(mean_loc_err, 3),
        }

    def generate_report_markdown(self) -> str:
        """Generates formatted Markdown benchmark table."""
        m = self.get_summary_metrics()
        return f"""
### MTMCT Synthetic Benchmark Evaluation Report

| Metric | Measured Value | Target Benchmark |
| :--- | :--- | :--- |
| **MOTA (Multi-Object Tracking Accuracy)** | `{m['mota'] * 100:.1f}%` | `> 85.0%` |
| **Precision** | `{m['precision'] * 100:.1f}%` | `> 90.0%` |
| **Recall** | `{m['recall'] * 100:.1f}%` | `> 88.0%` |
| **F1 Score** | `{m['f1_score'] * 100:.1f}%` | `> 89.0%` |
| **ID Switches (Cross-Camera / Intra)** | `{int(m['id_switches'])}` | `< 5 per 1k frames` |
| **Mean World Localization Error** | `{m['mean_localization_error_m']:.2f} meters` | `< 0.35 meters` |
| **Total Evaluated Ground Truth Targets** | `{int(m['total_ground_truth'])}` | `N/A` |
"""

    @staticmethod
    def _compute_iou(
        boxA: Tuple[float, float, float, float],
        boxB: Tuple[float, float, float, float]
    ) -> float:
        """Calculates Intersection over Union between two (x1, y1, x2, y2) boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea
        if unionArea <= 0.0:
            return 0.0
        return interArea / unionArea
