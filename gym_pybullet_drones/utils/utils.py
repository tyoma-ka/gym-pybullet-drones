"""General use functions.
"""
import os
import time
import argparse
from datetime import datetime

import numpy as np
import pybullet as p
import pyautogui
import json
from pathlib import Path

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


def is_close_to_obstacle_with_distance(raytraced_distances, nth_drone, threshold=1):
    for distance in raytraced_distances[nth_drone]:
        if distance < threshold:
            return True, distance
    return False, threshold


def get_closest_obstacle_distance(raytraced_distances, nth_drone):
    return min(raytraced_distances[nth_drone])


def get_obstacle_term(raytraced_directions, velocity_vector, nth_drone, threshold=1):
    if len(raytraced_directions[nth_drone]) == 0:
        return 0.0

    velocity_norm = np.linalg.norm(velocity_vector)
    if velocity_norm == 0:
        return 0.0

    velocity_unit = velocity_vector / velocity_norm
    dot_products = []

    for direction in raytraced_directions[nth_drone]:
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            continue
        direction_unit = direction / direction_norm
        dot = np.dot(velocity_unit, direction_unit)  # Cosine of angle between vectors
        dot_products.append(dot)

    if not dot_products:
        return 0.0

    aligned_toward_obstacles = [max(0, dp) for dp in dot_products]
    return np.mean(aligned_toward_obstacles)


def maximize_pybullet_mac():
    # this function clicks button to maximize pybullet gui window to the whole screen on my Mac :)
    current_x, current_y = pyautogui.position()
    green_button_x = 54
    green_button_y = 324

    # pyautogui.moveTo(green_button_x, green_button_y, duration=0.5)
    pyautogui.click(green_button_x, green_button_y)
    pyautogui.click(green_button_x, green_button_y)
    pyautogui.move(current_x, current_y)


def draw_target_circle(center, radius=0.2, color=None, segments=32, duration=0):
    """Draws a circle using line segments in PyBullet."""
    if color is None:
        color = [1, 0, 0]
    theta = np.linspace(0, 2 * np.pi, segments + 1)
    points = np.array([[center[0] + radius * np.cos(t),
                        center[1] + radius * np.sin(t),
                        center[2]] for t in theta])

    for i in range(segments):
        p.addUserDebugLine(points[i], points[i+1], lineColorRGB=color, lifeTime=duration, lineWidth=2)


def generate_ray_directions(drone_pos, drone_quat, laser_range=1.0, n_yaw=16, n_pitch=4):
    ray_from, ray_to = [], []

    yaw_angles = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)
    pitch_angles = np.linspace(-np.pi, np.pi / 3, n_pitch)  # From 0° to +60°

    _, _, yaw = p.getEulerFromQuaternion(drone_quat)

    rot_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    for pitch in pitch_angles:
        for yaw in yaw_angles:
            local_dir = np.array([
                laser_range * np.cos(pitch) * np.cos(yaw),
                laser_range * np.cos(pitch) * np.sin(yaw),
                laser_range * np.sin(pitch)
            ])
            world_dir = np.dot(rot_yaw, local_dir)
            ray_from.append(drone_pos)
            ray_to.append(drone_pos + world_dir)

    return np.array(ray_from), np.array(ray_to)


def load_training_config(path):
    with open(path, 'r') as f:
        config = json.load(f)

    if 'target_point' in config:
        config['target_point'] = np.array(config['target_point'])
    if 'initial_xyz' in config:
        config['initial_xyz'] = np.array(config['initial_xyz'])

    if 'collisions_pos_orn' in config:
        config['collisions_pos_orn'] = [
            (np.array(pos), np.array(orn)) for pos, orn in config['collisions_pos_orn']
        ]
    return config


def get_latest_results_folder(results_dir="results"):
    """
    Scans the given results directory and returns the full path to the latest
    'save-<timestamp>' folder based on its timestamp in the name.

    Parameters
    ----------
    results_dir : str
        Path to the base results directory (default: "results")

    Returns
    -------
    str or None
        Path to the most recent results subfolder, or None if not found
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # gym-pybullet-drones/
    results_dir = os.path.join(base_dir, results_dir)

    latest_folder = None
    latest_time = None

    for name in os.listdir(results_dir):
        if name.startswith("save-"):
            try:
                timestamp_str = name[5:]  # strip "save-"
                timestamp = datetime.strptime(timestamp_str, "%m.%d.%Y_%H.%M.%S")
                folder_path = os.path.join(results_dir, name)

                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_folder = folder_path
            except ValueError:
                continue  # Skip malformed names

    return latest_folder


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def compute_avoidance_point(drone_position, obstacle_vector, offset_distance):
    """
    Given a 3D vector pointing toward the closest obstacle,
    return a unit vector that is perpendicular to it in the x-y plane (90° to the right),
    and with the same z-coordinate.
    """
    x, y, _ = obstacle_vector

    # Compute right-perpendicular in x-y plane
    perp = np.array([y, -x])
    norm = np.linalg.norm(perp)
    if norm == 0:
        return np.array(drone_position)  # no avoidance if no direction

    # Scale to offset distance
    offset_xy = (perp / norm) * offset_distance

    # Construct target point: same z as current drone position
    target_point = np.array([
        drone_position[0] + offset_xy[0],
        drone_position[1] + offset_xy[1],
        drone_position[2]
    ])
    return target_point
