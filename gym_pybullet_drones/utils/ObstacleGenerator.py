import random

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.calculation import get_random_orientation, get_random_point_between_radii


def get_random_obstacle_pose_in_range(x_range=(-1, 1), y_range=(-1, 1), z=0.1):
    """
    Returns a random position and orientation (quaternion) for an obstacle.
    """
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)

    position = [x, y, z]
    orientation = get_random_orientation()

    return position, orientation


def get_random_obstacle_pose_between_radii(r0, r1):
    position = get_random_point_between_radii(r0, r1, False)
    orientation = get_random_orientation()
    return position, orientation


def generate_field_of_obstacles(n_rows, n_cols, d_between_rows, d_between_cols, position_jitter):
    obstacle_positions = []
    x_positions = []
    y_positions = []
    half_rows = ((n_rows - 1) / 2)
    half_cols = ((n_cols - 1) / 2)
    for i in range(n_rows):
        x_positions.append((half_rows - i) * d_between_rows)
    for i in range(n_cols):
        y_positions.append((half_cols - i) * d_between_cols)

    for x in x_positions:
        for y in y_positions:
            random_jitter_x = np.random.uniform(0, position_jitter) - position_jitter / 2
            random_jitter_y = np.random.uniform(0, position_jitter) - position_jitter / 2
            obstacle_positions.append(((x + random_jitter_x, y + random_jitter_y, 0.1), p.getQuaternionFromEuler([0, 0, 0])))

    return obstacle_positions


