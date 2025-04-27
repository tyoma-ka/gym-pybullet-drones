import numpy as np

from gym_pybullet_drones.utils.utils import randomize_initial_positions


class TrainingStateController:
    def __init__(self, target_point=None, initial_xyz=None, randomized_initial_xyz=True):
        if target_point is None:
            target_point = np.array([1, 1, 1])
        if initial_xyz is None:
            initial_xyz = np.array([0, 0, 0.125])

        self.target_point = target_point
        self.initial_xyz = initial_xyz
        self.randomized_initial_xyz = randomized_initial_xyz

    def set_target_point(self, target_point):
        self.target_point = target_point

    def get_target_point(self):
        return self.target_point

    def set_initial_xyz(self, initial_xyz):
        self.initial_xyz = initial_xyz

    def get_initial_xyz(self):
        return self.initial_xyz

    def get_and_update_initial_xyz(self, number_of_drones):
        result = self.initial_xyz
        if self.randomized_initial_xyz:
            self.randomize_initial_xyz(number_of_drones)
        return result

    def randomize_initial_xyz(self, number_of_drones):
        self.initial_xyz = randomize_initial_positions(number_of_drones)