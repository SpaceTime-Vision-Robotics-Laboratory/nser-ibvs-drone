import os
import pickle
from pathlib import Path

import numpy as np

from auto_follow.utils.path_manager import Paths


def read_pkl_file(path_to_file: str | Path):
    with open(path_to_file, 'rb') as f:
        data = pickle.load(f)
    return data


def scale_intrinsic_matrix_by_factor(k: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scales the intrinsic camera matrix by a given factor.

    :param k: 3x3 intrinsic camera matrix.
    :param scale_factor: How much to scale the intrinsic matrix (e.g., 0.5 for half resolution).
    :returns: The scaled 3x3 intrinsic matrix.
    """
    k_scaled = k.copy()
    k_scaled[0, 0] *= scale_factor  # Scale fx
    k_scaled[1, 1] *= scale_factor  # Scale fy
    k_scaled[0, 2] *= scale_factor  # Scale cx
    k_scaled[1, 2] *= scale_factor  # Scale cy

    return k_scaled


def scale_intrinsic_matrix_by_size(k: np.ndarray, original_size: tuple[int, int],
                                   new_size: tuple[int, int]) -> np.ndarray:
    """
    Scales a 3x3 camera intrinsic matrix `K` based on the change in resolution.

    :param k: 3x3 intrinsic camera matrix.
    :param original_size: (width, height) of the original image.
    :param new_size: (width, height) of the resized image.
    :returns: The scaled 3x3 intrinsic matrix.
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    k_scaled = k.copy()
    k_scaled[0, 0] *= scale_x  # fx
    k_scaled[1, 1] *= scale_y  # fy
    k_scaled[0, 2] *= scale_x  # cx
    k_scaled[1, 2] *= scale_y  # cy
    return k_scaled


def get_paths_for_param_res(param_res_dir: str | Path = Paths.CAMERA_REAL_FULL_ANAFI_4K_DIR):
    calibrate_path = os.path.join(param_res_dir, "calibrate.pkl")
    distortion_path = os.path.join(param_res_dir, "distortion.pkl")
    intrinsic_matrix_path = os.path.join(param_res_dir, "intrinsic_matrix.pkl")
    rotation_path = os.path.join(param_res_dir, "rotation_vectors.pkl")
    translation_path = os.path.join(param_res_dir, "translation_vectors.pkl")

    return calibrate_path, distortion_path, intrinsic_matrix_path, rotation_path, translation_path


def get_paths_for_sim_params_res(param_dir: str | Path = Paths.CAMERA_SIM_ANAFI_4k_DIR):
    intrinsic_matrix_path = os.path.join(param_dir, "intrinsic_matrix.pkl")
    intrinsic_matrix_half_path = os.path.join(param_dir, "intrinsic_matrix_half_size.pkl")

    return intrinsic_matrix_path, intrinsic_matrix_half_path


def infer_intrinsic_matrix(param_dir: str | Path) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    param_path = Path(param_dir)
    if "sim" in param_path.name:
        intrinsic_matrix_path, intrinsic_matrix_half_path = get_paths_for_sim_params_res(param_dir)
        return read_pkl_file(intrinsic_matrix_path), read_pkl_file(intrinsic_matrix_half_path)

    elif "real" in param_path.name:
        _, _, intrinsic_matrix_path, _, _ = get_paths_for_param_res(param_dir)
        return read_pkl_file(intrinsic_matrix_path)

    raise ValueError(f"Wrong camera parameters name format for ({param_path.name}) in \"{param_path}\"")
