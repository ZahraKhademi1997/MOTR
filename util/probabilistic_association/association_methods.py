
import numpy as np
import scipy.linalg
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_cxcyah
from cython_bbox import bbox_overlaps as bbox_ious
import torch
import cv2
import scipy
import lap
from scipy.spatial.distance import cdist
import time
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

# Assignement algorithm
# def linear_assignment(cost_matrix, thresh): # LAPJ algorithm
#     if cost_matrix.size == 0:
#         return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
#     matches, unmatched_a, unmatched_b = [], [], []
#     cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
#     for ix, mx in enumerate(x):
#         if mx >= 0:
#             matches.append([ix, mx])
#     unmatched_a = np.where(x < 0)[0]
#     unmatched_b = np.where(y < 0)[0]
#     matches = np.asarray(matches)
#     return matches, unmatched_a, unmatched_b

def linear_assignment(cost_matrix, thresh): # Hungarian algorithm
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    # Apply threshold to cost matrix
    masked_cost_matrix = np.ma.masked_greater(cost_matrix, thresh)
    
    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(masked_cost_matrix)
    
    # Filter out assignments above threshold
    valid_matches_mask = ~masked_cost_matrix[row_ind, col_ind].mask
    matches = np.column_stack((row_ind[valid_matches_mask], col_ind[valid_matches_mask]))
    
    unmatched_a = set(range(cost_matrix.shape[0])) - set(matches[:, 0])
    unmatched_b = set(range(cost_matrix.shape[1])) - set(matches[:, 1])
    
    return matches, list(unmatched_a), list(unmatched_b)


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious
def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[Track]
    :type btracks: list[Track]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix
