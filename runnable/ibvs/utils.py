import math
import numpy as np
import cv2

def e2h(v):
    """
    Convert from Euclidean to homogeneous form

    :param v: Euclidean vector or matrix
    :type v: array_like(n), ndarray(n,m)
    :return: homogeneous vector
    :rtype: ndarray(n+1,m)

    - If ``v`` is an N-vector, return an (N+1)-column vector where a value of 1 has
      been appended as the last element.
    - If ``v`` is a matrix (NxM), return a matrix (N+1xM), where each column has
      been appended with a value of 1, ie. a row of ones has been appended to the matrix.

    .. runblock:: pycon

        >>> from spatialmath.base import *
        >>> e2h([2, 4, 6])
        >>> e = np.c_[[1,2], [3,4], [5,6]]
        >>> e
        >>> e2h(e)

    .. note:: The result is always a 2D array, a 1D input results in a column vector.

    :seealso: e2h
    """
    if isinstance(v, np.ndarray) and len(v.shape) == 2:
        # dealing with matrix
        return np.vstack([v, np.ones((1, v.shape[1]))])
    else:
        raise ValueError("bad type")

def get_rectangle_sides(corners) -> tuple[float, float]:
    """
    :param corners: list of 4 (x, y) tuples representing rectangle corners
    :returns: Tuple of (short_side_length, long_side_length)
    """
    # Calculate only two adjacent sides
    x1, y1 = corners[0]
    x2, y2 = corners[1] 
    x3, y3 = corners[2]
    
    side1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    side2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    
    if side1 < side2:
        return side1, side2
    return side2, side1

def plot_bbox_keypoints(plotted_frame, keypoints):
    p1, p2, p3, p4 = keypoints

    cv2.circle(plotted_frame, p1, 5, (255, 0, 0), -1)
    cv2.circle(plotted_frame, p2, 5, (0,255,0), -1)

    cv2.circle(plotted_frame, p3, 5, (0,0,255), -1)
    cv2.circle(plotted_frame, p4, 5, (255, 255, 255), -1)

def plot_ellipse_keypoints(plotted_frame, keypoints):
    kp_m1, kp_m2, kp_M1, kp_M2 = keypoints

    cv2.line(plotted_frame, tuple(kp_m1.astype(np.int32)), tuple(kp_m2.astype(np.int32)), (255, 255, 0), 1)
    cv2.circle(plotted_frame, tuple(kp_m1.astype(np.int32)), 5, (255, 0, 0), -1)
    cv2.circle(plotted_frame, tuple(kp_m2.astype(np.int32)), 5, (0,255,0), -1)

    cv2.line(plotted_frame, tuple(kp_M1.astype(np.int32)), tuple(kp_M2.astype(np.int32)), (255, 255, 0), 1)
    cv2.circle(plotted_frame, tuple(kp_M1.astype(np.int32)), 5, (0,0,255), -1)
    cv2.circle(plotted_frame, tuple(kp_M2.astype(np.int32)), 5, (255, 255, 255), -1)

## ------------------------------------------
## stability
## ------------------------------------------

def check_stability(points_bbox, mean, std):
    sides = get_rectangle_sides(points_bbox)

    r = (1. * sides[1]) / sides[0]
    if (abs(r - mean) >= std):
        return False
    
    return True

## works on NOT ORIENTED BBOX
def check_stability_bbox_simple(points_bbox):
    sides = get_rectangle_sides(points_bbox)

    r = (1. * sides[1]) / sides[0]
    if (abs(r - 2.275) >= 0.207):
        return False
    
    return True

## works on ORIENTED BBOX
def check_stability_bbox_oriented(points_bbox):
    sides = get_rectangle_sides(points_bbox)

    r = (1. * sides[1]) / sides[0]
    if (abs(r - 2.401) >= 0.18):
        return False
    
    return True

## ------------------------------------------

def compute_distance(p1, p2):
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)

    return np.linalg.norm(p1_arr - p2_arr)
