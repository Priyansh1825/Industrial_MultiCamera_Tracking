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

    area_intersection = np.prod(np.clip(br - tl, a_min=0, a_max=None), axis=2)
    area_a = np.prod(boxes_a[:, 2:] - boxes_a[:, :2], axis=1)
    area_b = np.prod(boxes_b[:, 2:] - boxes_b[:, :2], axis=1)
    area_union = area_a[:, None] + area_b[None, :] - area_intersection

    return np.clip(area_intersection / np.maximum(area_union, 1e-6), 0.0, 1.0)


def iou_distance(atracks: list, btracks: list) -> np.ndarray:
    """
    Computes cost matrix based on 1.0 - IoU.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = bbox_ious(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1.0 - _ious
    return cost_matrix


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
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b
