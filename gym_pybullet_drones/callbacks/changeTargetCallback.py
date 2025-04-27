from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController
from gym_pybullet_drones.utils.utils import randomize_target_point


class ChangeTargetOnRewardCallback(BaseCallback):
    def __init__(self, reward_threshold=800, training_state_controller: TrainingStateController=None, verbose=1):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.training_state_controller = training_state_controller
        self.current_episode_reward = 0
        self.counter = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.counter += 1
        if self.locals['dones'][0]:
            env = self.training_env.envs[0].unwrapped

            # if self.current_episode_reward > self.reward_threshold:
            self.training_state_controller.set_target_point(np.copy(randomize_target_point()))
                # print(f"Target point changed to {self.target_point_controller.get_target_point()}")

            # print(f"Reward: {self.current_episode_reward}, steps: {self.counter}, target: {current_target}")
            self.counter = 0
            self.current_episode_reward = 0

        return True
