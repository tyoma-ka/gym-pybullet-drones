import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.utils import randomize_initial_positions, generate_random_obstacle_pose


class TrainingStateController:
    def __init__(self,
                 target_point=None,
                 initial_xyz=None,
                 randomized_initial_xyz=True,
                 collisions_pos_orn=None,
                 randomized_collisions=True,
                 number_of_collisions=1):
        if collisions_pos_orn is None:
            collisions_pos_orn = []
        if target_point is None:
            target_point = np.array([1, 1, 1])
        if initial_xyz is None:
            initial_xyz = np.array([[0, 0, 0.125]])

        self.collisions_pos_orn = []
        for pos, orn in collisions_pos_orn:
            orn_rad = (np.radians(orn[0]), np.radians(orn[1]), np.radians(orn[2]))
            self.collisions_pos_orn.append((pos, p.getQuaternionFromEuler(orn_rad)))

        self.target_point = target_point
        self.initial_xyz = initial_xyz
        self.randomized_initial_xyz = randomized_initial_xyz
        self.randomized_collisions = randomized_collisions
        self.number_of_collisions = number_of_collisions

    def set_target_point(self, target_point):
        self.target_point = target_point

    def get_target_point(self):
        return self.target_point

    def set_initial_xyz(self, initial_xyz):
        self.initial_xyz = initial_xyz

    def get_initial_xyz(self):
        return self.initial_xyz

    def get_and_update_initial_xyz(self, number_of_drones, obstacles_aabbs):
        result = self.initial_xyz
        if self.randomized_initial_xyz:
            self.randomize_initial_xyz(number_of_drones, obstacles_aabbs)
        return result

    def randomize_initial_xyz(self, number_of_drones, obstacles_aabbs):
        self.initial_xyz = randomize_initial_positions(number_of_drones, obstacles_aabbs)

    def get_number_of_collisions(self):
        return self.number_of_collisions

    def get_and_update_collisions_pos_orn(self):
        result = []
        if self.randomized_collisions:
            for i in range(self.number_of_collisions):
                pos, orn = generate_random_obstacle_pose()
                result.append((pos, orn))
        else:
            result = self.collisions_pos_orn
        return result
