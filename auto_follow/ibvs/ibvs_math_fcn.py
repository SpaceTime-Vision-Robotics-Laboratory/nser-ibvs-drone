import math

import cv2
import numpy as np


def e2h(v: np.ndarray | list[float]) -> np.ndarray:
    """
    Convert from Euclidean to homogeneous form

    :param v: Euclidean vector or matrix
    :returns: Homogeneous vector
    """
    max_num_dim = 2

    if isinstance(v, np.ndarray) and len(v.shape) == max_num_dim:
        return np.vstack([v, np.ones((1, v.shape[1]))])
    else:
        raise ValueError("bad type")


def get_rectangle_sides(corners: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Get the sides of the rectangle from the corners

    :param corners: list of 4 (x, y) tuples representing rectangle corners
    :returns: Tuple of (short_side_length, long_side_length)
    """
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    x3, y3 = corners[2]

    side1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    side2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

    if side1 < side2:
        return side1, side2
    return side2, side1


def plot_bbox_keypoints(plotted_frame: np.ndarray, keypoints: list[tuple[int, int]] | list[tuple[int, ...]]) -> None:
    """
    Plot the bbox keypoints on the frame
    """
    p1, p2, p3, p4 = keypoints

    cv2.circle(plotted_frame, p1, 5, (255, 0, 0), -1)
    cv2.circle(plotted_frame, p2, 5, (0, 255, 0), -1)

    cv2.circle(plotted_frame, p3, 5, (0, 0, 255), -1)
    cv2.circle(plotted_frame, p4, 5, (255, 255, 255), -1)


def plot_ellipse_keypoints(plotted_frame: np.ndarray, keypoints: list[tuple[int, int]]) -> None:
    """
    Plot the ellipse keypoints on the frame
    """
    kp_min1, kp_min2, kp_max1, kp_max2 = keypoints

    cv2.line(plotted_frame, tuple(kp_min1.astype(np.int32)), tuple(kp_min2.astype(np.int32)), (255, 255, 0), 1)
    cv2.circle(plotted_frame, tuple(kp_min1.astype(np.int32)), 5, (255, 0, 0), -1)
    cv2.circle(plotted_frame, tuple(kp_min2.astype(np.int32)), 5, (0, 255, 0), -1)

    cv2.line(plotted_frame, tuple(kp_max1.astype(np.int32)), tuple(kp_max2.astype(np.int32)), (255, 255, 0), 1)
    cv2.circle(plotted_frame, tuple(kp_max1.astype(np.int32)), 5, (0, 0, 255), -1)
    cv2.circle(plotted_frame, tuple(kp_max2.astype(np.int32)), 5, (255, 255, 255), -1)


def check_stability(points_bbox: list[tuple[float, float]], mean: float, std: float) -> bool:
    """
    Check if the bbox is stable
    """
    sides = get_rectangle_sides(points_bbox)

    r = (1. * sides[1]) / sides[0]
    print(f"{r=}")
    if abs(r - mean) >= std:
        return False

    return True

## TODO these two below are for the simple scene in the simulator (with the yellow car from Sphinx)
def check_stability_bbox_simple(points_bbox: list[tuple[float, float]]) -> bool:
    """
    Check if the bbox is stable that works on Simple BBOX(not oriented)
    """
    sides = get_rectangle_sides(points_bbox)

    stability_ratio = 2.275
    stability_std = 0.207

    r = (1. * sides[1]) / sides[0]
    if abs(r - stability_ratio) >= stability_std:
        return False

    return True


def check_stability_bbox_oriented(points_bbox: list[tuple[float, float]]) -> bool:
    """
    Check if the bbox is stable that works on Oriented BBOX
    """
    sides = get_rectangle_sides(points_bbox)

    stability_ratio = 2.401
    stability_std = 0.18

    r = (1. * sides[1]) / sides[0]
    if abs(r - stability_ratio) >= stability_std:
        return False

    return True


def compute_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    p1_arr = np.array(p1)
    p2_arr = np.array(p2)

    return np.linalg.norm(p1_arr - p2_arr)
