"""General use functions.
"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import pyautogui
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")


def _get_random_point_exclude_obstacles(obstacles_aabbs: list):
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


def randomize_target_point(obstacles_aabbs: list):
    return _get_random_point_exclude_obstacles(obstacles_aabbs)


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

    return min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z


def randomize_initial_positions(number_of_drones, obstacles_aabbs):
    return (np.vstack(_get_random_point_exclude_obstacles(obstacles_aabbs))
            .transpose()
            .reshape(number_of_drones, 3))


def get_norm_path(path: str):
    ret_path = os.path.join(
        os.path.dirname(__file__),
        path
    )
    return os.path.normpath(ret_path)


def is_close_to_obstacle(raytraced_distances, nth_drone, threshold=1):
    for distance in raytraced_distances[nth_drone]:
        if distance < threshold:
            return True
    return False


def generate_random_obstacle_pose(x_range=(-1, 1), y_range=(-1, 1), z=0.1, yaw_range=(0, 360)):
    """
    Returns a random position and orientation (quaternion) for an obstacle.
    """
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    yaw_deg = np.random.uniform(*yaw_range)
    yaw_rad = np.radians(yaw_deg)

    position = [x, y, z]
    orientation = p.getQuaternionFromEuler([0, 0, yaw_rad])

    return position, orientation


def maximize_pybullet_mac():
    # this function clicks button to maximize pybullet gui window to the whole screen on my Mac :)
    current_x, current_y = pyautogui.position()
    green_button_x = 54
    green_button_y = 324

    # pyautogui.moveTo(green_button_x, green_button_y, duration=0.5)
    pyautogui.click(green_button_x, green_button_y)
    pyautogui.move(current_x, current_y)
