import os
import time

import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3.common.evaluation import evaluate_policy

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')
DEFAULT_OUTPUT_FOLDER = 'results'


def run_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    model = PPO.load(path)
    test_env = HoverAviary(gui=True,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=True)
    test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=DEFAULT_OUTPUT_FOLDER,
                    colab=False
                    )
    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

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
        if DEFAULT_OBS == ObservationType.KIN:

            logger.log(drone=0,
                       timestamp=i / test_env.CTRL_FREQ,
                       state=np.hstack([obs2[0:3],
                                        np.zeros(4),
                                        obs2[3:15],
                                        act2
                                        ]),
                       control=np.zeros(12)
                       )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if DEFAULT_OBS == ObservationType.KIN:
        logger.plot()


run_model('results/save-03.31.2025_19.33.46/best_model.zip')