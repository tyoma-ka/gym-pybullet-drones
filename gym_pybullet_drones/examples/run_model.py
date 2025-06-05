import os
import time

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

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
from gym_pybullet_drones.utils.constants import aviary_map


def run_model(open_last: bool, model_type: str, path: str | None):
    folder_path = path
    if open_last:
        folder_path = get_latest_results_folder()
        folder_path += "/final_model.zip" if model_type == "final" else "/best_model.zip"

    if not os.path.isfile(folder_path):
        raise FileNotFoundError(folder_path)

    cfg = load_training_config(get_norm_path("../gui_config.json"))
    training_state_controller = TrainingStateController(**cfg)

    model = PPO.load(folder_path)

    aviary_key = training_state_controller.aviary
    if aviary_key not in aviary_map:
        raise ValueError(f"Unknown aviary type: {aviary_key}")

    aviary_class = aviary_map[aviary_key]

    test_env = aviary_class(
        gui=training_state_controller.gui,
        obs=training_state_controller.observation_type,
        act=training_state_controller.action_type,
        training_state_controller=training_state_controller,
        initial_xyzs=np.array([1, 1, 1]),
        record=True
    )

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=training_state_controller.output_folder,
                    colab=False
                    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:",
              truncated)
        # print(reward)
        if training_state_controller.observation_type == ObservationType.KIN:

            logger.log(drone=0,
                       timestamp=i / test_env.CTRL_FREQ,
                       state=np.hstack([obs2[9:12],
                                        obs2[0:3],
                                        np.zeros(1),
                                        obs2[3:15],
                                        act2,
                                        reward
                                        ]),
                       control=np.zeros(12)
                       )
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs, info = test_env.reset(seed=42, options={})
    test_env.close()

    logger.plot_distances(save_path=os.path.dirname(folder_path) + "/ep_distance_plot.png")
    logger.plot_velocity(save_path=os.path.dirname(folder_path) + "/ep_velocity_plot.png")
    logger.plot_orientation(save_path=os.path.dirname(folder_path) + "/ep_orientation_plot.png")
    logger.plot_reward(save_path=os.path.dirname(folder_path) + "/ep_reward_plot.png")


run_model(True, "best", 'results/save-06.01.2025_10.03.57/best_model.zip')
