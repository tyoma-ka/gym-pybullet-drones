import random
import numpy as np

from gym_pybullet_drones.utils.calculation import get_random_point_between_radii


def get_random_point_in_range(x_from, x_to, y_from, y_to, z_from, z_to):
    x = np.random.uniform(x_from, x_to)
    y = np.random.uniform(y_from, y_to)
    z = np.random.uniform(z_from, z_to)
    return np.array([x, y, z])


def randomize_initial_positions(obstacles_aabbs):
    # return _get_random_point_exclude_obstacles(obstacles_aabbs)
    return get_random_point_between_radii(0.5, 1, True)


def get_opposite_point(point):
    return np.array([-point[0], -point[1], 1 - point[2]])


def randomize_target_point(obstacles_aabbs: list):
    # return _get_random_point_exclude_obstacles(obstacles_aabbs)
    return get_random_point_between_radii(0.5, 1, True)



