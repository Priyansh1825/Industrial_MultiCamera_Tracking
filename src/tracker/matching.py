"""
Bipartite matching and geometric IoU distance matrix utilities for track association.
"""
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_ious(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise IoU between two sets of bounding boxes in format [x1, y1, x2, y2].
    boxes_a: (N, 4)
    boxes_b: (M, 4)
    Returns: (N, M) matrix with IoU values between 0.0 and 1.0.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.ascontiguousarray(boxes_a, dtype=float)
    boxes_b = np.ascontiguousarray(boxes_b, dtype=float)

    # Compute overlapping top-left and bottom-right corners
    tl = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    br = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])

    wh = np.clip(br - tl, a_min=0, a_max=None)
    area_intersection = wh[:, :, 0] * wh[:, :, 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    area_union = area_a[:, None] + area_b[None, :] - area_intersection

    iou = np.clip(area_intersection / np.maximum(area_union, 1e-6), 0.0, 1.0)
    return iou.astype(np.float32)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Computes cost matrix based on 1.0 - IoU.
    Handles both lists of STrack objects and raw numpy arrays of boxes.
    """
    # Check if inputs are already box arrays (ndarray)
    atracks_is_boxes = len(atracks) > 0 and isinstance(atracks[0], np.ndarray)
    btracks_is_boxes = len(btracks) > 0 and isinstance(btracks[0], np.ndarray)

    if atracks_is_boxes and btracks_is_boxes:
        atlbrs = np.asarray(atracks)
        btlbrs = np.asarray(btracks)
    elif atracks_is_boxes or btracks_is_boxes:
        raise ValueError("Both atracks and btracks must be either STrack lists or box arrays, not mixed")
    else:
        atlbrs = np.array([track.tlbr for track in atracks], dtype=np.float32)
        btlbrs = np.array([track.tlbr for track in btracks], dtype=np.float32)

    # Validate boxes have positive area
    if len(atlbrs) > 0:
        invalid_a = (atlbrs[:, 2] <= atlbrs[:, 0]) | (atlbrs[:, 3] <= atlbrs[:, 1])
        if np.any(invalid_a):
            atlbrs[invalid_a, 2:] = atlbrs[invalid_a, :2] + 1.0
    if len(btlbrs) > 0:
        invalid_b = (btlbrs[:, 2] <= btlbrs[:, 0]) | (btlbrs[:, 3] <= btlbrs[:, 1])
        if np.any(invalid_b):
            btlbrs[invalid_b, 2:] = btlbrs[invalid_b, :2] + 1.0

    _ious = bbox_ious(atlbrs, btlbrs)
    cost_matrix = 1.0 - _ious
    return cost_matrix.astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves optimal bipartite matching problem using the Hungarian algorithm.
    Returns:
    - matches: List of (row_idx, col_idx) matched pairs
    - unmatched_a: List of unmatched row indices
    - unmatched_b: List of unmatched column indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]

    return matches, unmatched_a, unmatched_b