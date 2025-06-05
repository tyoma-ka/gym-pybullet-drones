import os
import time

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.PositionFlyToAviary import PositionFlyToAviary
from gym_pybullet_drones.envs.PositionObstacleFlyToAviary import PositionObstacleFlyToAviary
from gym_pybullet_drones.envs.OrientationFlyToAviary import OrientationFlyToAviary
from gym_pybullet_drones.envs.OrientationObstacleFlyToAviary import OrientationObstacleFlyToAviary
from gym_pybullet_drones.utils.EvaluatingLogger import EvaluatingLogger
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import PPO, SAC
from gym_pybullet_drones.utils.utils import sync, str2bool, load_training_config, get_norm_path, \
    get_latest_results_folder
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.constants import aviary_map, aviary_names


def evaluate_model(open_last: bool, model_type: str, path: str | None):
    ev = EvaluatingLogger()

    # path = get_norm_path(path)
    folder_path = os.path.dirname(path)
    if open_last:
        folder_path = get_latest_results_folder()
        path = folder_path
        path += "/final_model.zip" if model_type == "final" else "/best_model.zip"

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    cfg = load_training_config(get_norm_path("../eval_config.json"))
    training_state_controller = TrainingStateController(**cfg)

    model = PPO.load(path)

    aviary_key = training_state_controller.aviary
    if aviary_key not in aviary_map:
        raise ValueError(f"Unknown aviary type: {aviary_key}")

    aviary_class = aviary_map[aviary_key]
    aviary_name = aviary_names[aviary_class]

    test_env_nogui = aviary_class(obs=training_state_controller.observation_type,
                                  act=training_state_controller.action_type,
                                  gui=False,
                                  training_state_controller=training_state_controller)

    n_episodes = 1000
    all_rewards = []
    all_distances = []
    all_velocities = []
    all_orientations = []
    all_hits = []
    all_reaches = []
    all_distances_to_obstacle = []

    for ep in tqdm(range(n_episodes)):
        obs, info = test_env_nogui.reset(seed=42, options={})
        rewards = []
        distances = []
        velocities = []
        orientations = []
        hits = 0
        reaches = 0
        distances_to_obstacle = []

        for i in range((test_env_nogui.EPISODE_LEN_SEC + 2) * test_env_nogui.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True
                                            )
            obs, reward, terminated, truncated, info = test_env_nogui.step(action)
            if terminated:
                break
            rewards.append(reward)

            hit_obstacle = info["hit_obstacle"]
            reached_target = info["reached_target"]
            distance_to_target = info["distance_to_target"]
            velocity = info["velocity"]
            orientation = info["orientation"]
            distance_to_obstacle = info["distance_to_obstacle"]

            distances.append(distance_to_target)
            velocities.append(velocity)
            orientations.append(orientation)
            distances_to_obstacle.append(distance_to_obstacle)

            if hit_obstacle:
                hits += 1

            if reached_target:
                reaches += 1

        all_rewards.append(rewards)
        all_distances.append(distances)
        all_velocities.append(velocities)
        all_orientations.append(orientations)
        all_hits.append(hits)
        all_reaches.append(reaches)
        all_distances_to_obstacle.append(distances_to_obstacle)

    reward_means, reward_stds = get_mean_and_std(all_rewards)
    distance_means, distance_stds = get_mean_and_std(all_distances)
    velocity_means, velocity_stds = get_mean_and_std(all_velocities)
    orientation_means, orientation_stds = get_mean_and_std(all_orientations)
    distances_to_obstacle_means, distances_to_obstacle_stds = get_mean_and_std(all_distances_to_obstacle)

    ev.plot_two_graphs_side_by_side(
        np.arange(len(reward_means)),
        reward_means,
        reward_stds,
        "Timestep",
        "Reward",
        f"In-Episode Reward Progression ({n_episodes} episodes) ({aviary_name})",
        "Reward",

        np.arange(len(distance_means)),
        distance_means,
        distance_stds,
        "Timestep",
        "Distance to the target",
        f"In-Episode Distance to the Target Progression ({n_episodes} episodes) ({aviary_name})",
        "Distance",

        True,
        "stats_side_by_side_reward_dist.png",
        folder_path
    )

    ev.plot_two_graphs_side_by_side(
        np.arange(len(velocity_means)),
        velocity_means,
        velocity_stds,
        "Timestep",
        "Velocity",
        f"In-Episode Velocity Progression ({n_episodes} episodes) ({aviary_name})",
        "Velocity",

        np.arange(len(orientation_means)),
        orientation_means,
        orientation_stds,
        "Timestep",
        "Orientation magnitude",
        f"In-Episode Orientation Magnitude Progression ({n_episodes} episodes) ({aviary_name})",
        "Orientation magnitude",

        True,
        "stats_side_by_side_vel_orient.png",
        folder_path)

    ev.plot_graph(
        np.arange(len(distances_to_obstacle_means)),
        distances_to_obstacle_means,
        distances_to_obstacle_stds,
        "Timestep",
        "Distance to the closest obstacle",
        f"In-Episode Distance to the Closest obstacle Progression ({n_episodes} episodes) ({aviary_name})",
        "Distance to the Closest obstacle",

        "stats_dist_to_obstacle.png",
        True,
        folder_path
    )

    n_episodes_with_hit = sum(1 for x in all_hits if x != 0)
    n_episodes_with_reach = sum(1 for x in all_reaches if x != 0)

    print(f"From {n_episodes}, in {n_episodes - n_episodes_with_hit} episodes drone didn't hit the obstacle")
    print(f"From {n_episodes}, in {n_episodes_with_reach} episodes drone reached target")


def get_mean_and_std(all_values):
    max_values_len = max(len(r) for r in all_values)

    # padded if there is an inconsistency in episode lengths
    all_values_padded = np.array([r + [r[-1]] * (max_values_len - len(r)) for r in all_values])

    values_means = np.mean(np.array(all_values_padded), axis=0)
    values_stds = np.std(np.array(all_values_padded), axis=0)

    return values_means, values_stds


evaluate_model(True, "best", 'results/save-06.01.2025_18.54.18/best_model.zip')
