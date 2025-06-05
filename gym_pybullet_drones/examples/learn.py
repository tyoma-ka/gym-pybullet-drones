"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.callbacks.LogEvalCallback import LogEvalCallback
from gym_pybullet_drones.callbacks.changeTargetCallback import ChangeTargetOnRewardCallback
from gym_pybullet_drones.envs.PositionFlyToAviary import PositionFlyToAviary
from gym_pybullet_drones.envs.PositionObstacleFlyToAviary import PositionObstacleFlyToAviary
from gym_pybullet_drones.envs.OrientationFlyToAviary import OrientationFlyToAviary
from gym_pybullet_drones.envs.OrientationObstacleFlyToAviary import OrientationObstacleFlyToAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.TrainingLogger import TrainingLogger
from gym_pybullet_drones.utils.constants import aviary_map
from gym_pybullet_drones.utils.utils import sync, str2bool, load_training_config, get_norm_path
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController


def run():
    cfg = load_training_config(get_norm_path("../training_config.json"))
    training_state_controller = TrainingStateController(**cfg)

    filename = os.path.join(training_state_controller.output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    aviary_key = training_state_controller.aviary
    if aviary_key not in aviary_map:
        raise ValueError(f"Unknown aviary type: {aviary_key}")

    aviary_class = aviary_map[aviary_key]

    train_env = make_vec_env(aviary_class,
                             env_kwargs=dict(
                                 obs=training_state_controller.observation_type,
                                 act=training_state_controller.action_type,
                                 gui=training_state_controller.gui,
                                 training_state_controller=training_state_controller,
                                 ),
                             n_envs=1,
                             seed=0,
                             )
    eval_env = aviary_class(
        obs=training_state_controller.observation_type,
        act=training_state_controller.action_type,
        training_state_controller=training_state_controller,
        )

    #### Check the environment's spaces ########################
    print('[INFO] Aviary:', aviary_class)
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=1,

        # Training Hyperparameters
        learning_rate=3e-4,  # Adjust if needed
        n_steps=2048,  # Increase for more stable updates
        batch_size=64,  # Increase if learning is unstable
        gamma=0.99,  # Discount factor for long-term rewards

        # Policy & Exploration
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,  # PPO clipping (reduces catastrophic updates)

        # Normalize & Scale Rewards
        normalize_advantage=True,
        use_sde=True,  # Stochastic policy (improves exploration)
        policy_kwargs=dict(net_arch=[64, 64])
    )

    # model = SAC(
    #     "MlpPolicy",  # Use MLP policy (works well for low-dimensional input)
    #     env=train_env,  # Your drone environment
    #     verbose=1,
    #
    #     # === Training Hyperparameters ===
    #     learning_rate=3e-4,  # Common and stable; can try 1e-4 or 1e-3 if needed
    #     buffer_size=1000000,  # Big replay buffer (default is 1M)
    #     batch_size=256,  # Higher batch size helps stabilize SAC
    #     tau=0.005,  # Target smoothing coefficient (for stability)
    #     gamma=0.99,  # Discount factor for long-term rewards
    #
    #     # === SAC-specific Parameters ===
    #     train_freq=1,  # Update every step
    #     gradient_steps=1,  # One update per step (can try >1 for faster learning)
    #     ent_coef="auto",  # Let SAC automatically tune the entropy regularization
    #     learning_starts=10000,  # Steps before training starts
    #
    #     # === Policy Architecture ===
    #     policy_kwargs=dict(
    #         net_arch=[256, 256]  # Bigger network to handle more complex dynamics
    #     )
    # )

    #### Target cumulative rewards (problem-dependent) ##########

    target_reward = 40000
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1)

    training_logger = TrainingLogger("PPO")

    eval_callback = LogEvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=int(10_000),
        deterministic=True,
        render=False,
        n_eval_episodes=50,
        training_logger=training_logger
    )

    model.learn(total_timesteps=int(training_state_controller.total_timesteps),
                callback=eval_callback,
                log_interval=100)

    training_logger.plot(filename)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))


if __name__ == '__main__':
    run()
