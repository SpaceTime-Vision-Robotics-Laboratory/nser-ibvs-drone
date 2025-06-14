import matplotlib.pyplot as plt
import numpy as np

from auto_follow.ibvs.ibvs_math_fcn import e2h


class ImageBasedVisualServo:
    def __init__(
            self,
            camera_intrensic: np.ndarray,
            goal_points: list[tuple[int, int]],
            lambda_factor: float = 0.30,
            estimated_depth: float = 1.,
            verbose: bool = False
    ):
        self.K = camera_intrensic
        self.Kinv = np.linalg.inv(self.K)

        self.lambda_factor_val = lambda_factor
        self.lambda_factor = np.diag([lambda_factor * 2.5, lambda_factor * 2.5, lambda_factor * 1.15])
        self.lambda_factor_low = np.diag([0.5, 0.5, 0.1])

        self.goal_points = goal_points
        self.goal_points_flatten = np.hstack(goal_points)

        self.current_points = None
        self.current_points_normalized = None
        self.current_points_flatten = None

        self.jcond_values = []
        self.err_values = []
        self.err_uv_values = []

        self.Z = estimated_depth
        self.verbose = verbose

        self.err_threshold = 60

    def compute_normalized_image_plane_coordinates(self, points: list[tuple[int, int]]) -> np.ndarray:
        points_normalized = []
        for p in points:
            p_array = np.array([[p[0]], [p[1]]])
            xy = self.Kinv @ e2h(p_array)
            x = xy[0, 0]
            y = xy[1, 0]
            points_normalized.append((x, y))

        points_normalized = np.array(points_normalized)
        points_normalized = np.hstack(points_normalized)

        return points_normalized

    def set_current_points(self, current_points: list[tuple[int, int]]) -> None:
        self.current_points = current_points
        self.current_points_flatten = np.hstack(self.current_points)

    def compute_depths(self, pixels: np.ndarray) -> np.ndarray:
        """
        Estimate per-point depth based on image geometry.
        """
        depths = []
        for u, v in pixels:
            pixel_homog = np.array([u, v, 1.0])
            norm_coords = self.Kinv @ pixel_homog  # Gives [x_n, y_n, 1]
            x_n, y_n = norm_coords[0], norm_coords[1]
            z_i = self.Z * np.sqrt(x_n ** 2 + y_n ** 2 + 1)
            depths.append(z_i)

        return np.array(depths)

    def compute_interaction_matrix(self) -> np.ndarray:
        """
        Compute the interaction matrix
        """
        jacobian_matrix = np.empty((0, 3))
        depths = self.compute_depths(self.current_points)

        for depth, p in zip(depths, self.current_points):
            p_array = np.array([[p[0]], [p[1]]])

            xy = self.Kinv @ e2h(p_array)
            x = xy[0, 0]
            y = xy[1, 0]

            jacobian_point = self.K[:2, :2] @ np.array([
                [-1 / depth, 0, y],
                [0, -1 / depth, -x]
            ])

            jacobian_matrix = np.vstack([jacobian_matrix, jacobian_point])

            if self.verbose:
                print(f"J_point: {jacobian_point}")

        return jacobian_matrix

    def compute_velocities(self, verbose: bool = False) -> np.ndarray:
        jacobian_matrix = self.compute_interaction_matrix()

        jcond = np.linalg.cond(jacobian_matrix)
        self.jcond_values.append(jcond)

        jacobian_matrix_pinv = np.linalg.pinv(jacobian_matrix)

        err_uv = self.goal_points_flatten - self.current_points_flatten
        self.err_uv_values.append(np.linalg.norm(err_uv))

        if self.err_uv_values[-1] < self.err_threshold:
            lambda_factor = self.lambda_factor_low
        else:
            lambda_factor = self.lambda_factor

        if isinstance(lambda_factor, np.ndarray):
            vel = lambda_factor @ jacobian_matrix_pinv @ err_uv
        else:
            vel = lambda_factor * jacobian_matrix_pinv @ err_uv

        if verbose:
            print("-" * 25)
            print(f"J cond: {jcond}")
            print(f"Current points flat: {self.current_points_flatten}")
            print(f"Goal points flat: {self.goal_points_flatten}")
            print(f"Error uv: {err_uv} | error norm: {self.err_uv_values[-1]}")
            print(f"Velocity: {vel}")
            print("-" * 25)

        logs = {
            "jacobian_matrix": jacobian_matrix.tolist(),
            "jcond": jcond,
            "current_points_flatten": self.current_points_flatten,
            "goal_points_flatten": self.goal_points_flatten,
            "err_uv": err_uv,
            "velocity": vel
        }

        return vel, logs

    def plot_values(self):
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(range(len(self.jcond_values)), self.jcond_values, 'b-o')
        ax1.set_title('J_cond')
        ax1.set_ylabel('value')
        ax1.grid(True)

        ax2.plot(range(len(self.err_uv_values)), self.err_uv_values, 'r-o')
        ax2.set_title('error')
        ax2.set_xlabel('idx')
        ax2.set_ylabel('err norm')
        ax2.grid(True)

        plt.savefig("_check_errs.jpg", bbox_inches='tight')
