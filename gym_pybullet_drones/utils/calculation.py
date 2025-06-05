import pybullet as p
import numpy as np


def get_random_point_between_radii(r0, r1, randomize_z=False):
    if r0 >= r1:
        raise ValueError("r0 must be less than r1")

    radius = np.sqrt(np.random.uniform(r0 ** 2, r1 ** 2))
    angle = np.random.uniform(0, 2 * np.pi)

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    if randomize_z:
        z = np.random.uniform(0.1, 1)
    else:
        z = 0.1

    return np.array([x, y, z])


def get_random_point_exclude_obstacles(obstacles_aabbs: list):
    while True:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)
        position = (x, y, z)

        for aabb in obstacles_aabbs:
            if _position_inside_of_range(position, aabb):
                break
        else:
            return np.array([x, y, z])


def _position_inside_of_range(position, target_range):
    if len(position) != 3 or len(target_range) != 2 or len(target_range[0]) != 3 or len(target_range[1]) != 3:
        raise IndexError("position or target range are incorrect")

    x = position[0]
    y = position[1]
    z = position[2]
    min_bound = target_range[0]
    max_bound = target_range[1]
    min_x, min_y, min_z = min_bound
    max_x, max_y, max_z = max_bound

    _RANGE_OFFSET = 0.2

    min_x -= _RANGE_OFFSET
    min_y -= _RANGE_OFFSET
    min_z -= _RANGE_OFFSET

    max_x += _RANGE_OFFSET
    max_y += _RANGE_OFFSET
    max_z += _RANGE_OFFSET

    return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z


def get_random_orientation():
    yaw_deg = np.random.uniform(0, 360)
    yaw_rad = np.radians(yaw_deg)

    return p.getQuaternionFromEuler([0, 0, yaw_rad])
