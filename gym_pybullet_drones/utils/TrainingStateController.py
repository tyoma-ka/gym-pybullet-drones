import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.PositionGenerator import randomize_initial_positions, \
    randomize_target_point, get_random_point_in_range, get_opposite_point
from gym_pybullet_drones.utils.ObstacleGenerator import generate_field_of_obstacles, \
    get_random_obstacle_pose_between_radii
from gym_pybullet_drones.utils.enums import ExperimentType, ObservationType, ActionType
from gym_pybullet_drones.utils.utils import get_norm_path


class TrainingStateController:
    def __init__(self,
                 target_point=None,
                 randomized_target_point=True,
                 initial_xyz=None,
                 randomized_initial_xyz=True,
                 collisions_pos_orn=None,
                 randomized_collisions=True,
                 number_of_collisions=1,
                 show_gizmos=True,
                 show_lidar_rays=False,
                 show_velocity_vector=False,
                 plot_graph=True,
                 experiment=0,
                 aviary="HoverAviary",
                 episode_length=8,
                 total_timesteps=1_000_000,
                 laser_range=0.2,
                 observation_type=None,
                 action_type=None,
                 default_output_folder="",
                 default_gui=False):
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
        self.randomized_target_point = randomized_target_point
        self.randomized_initial_xyz = randomized_initial_xyz
        self.randomized_collisions = randomized_collisions
        self.number_of_collisions = number_of_collisions
        self.show_gizmos = show_gizmos
        self.show_lidar_rays = show_lidar_rays
        self.show_velocity_vector = show_velocity_vector
        self.plot_graph = plot_graph
        self.experiment = ExperimentType(experiment)
        self.aviary = aviary
        self.episode_length = episode_length
        self.total_timesteps = total_timesteps
        self.observation_type = ObservationType(observation_type)
        self.action_type = ActionType(action_type)
        self.output_folder = default_output_folder
        self.gui = default_gui
        self.laser_range = laser_range

        self.previous_xyz = None
        self.local_target_point = target_point

    def set_target_point(self, target_point):
        self.target_point = target_point
        self.show_target_point_circle()

    def set_local_target_point(self, local_target_point):
        self.local_target_point = local_target_point
        # if local_target_point is not None:
        #     self.show_local_target_point_circle()

    def get_target_point(self):
        if self.local_target_point is not None:
            return self.local_target_point
        return self.target_point

    def set_initial_xyz(self, initial_xyz):
        self.initial_xyz = initial_xyz

    def get_initial_xyz(self):
        return self.initial_xyz

    def get_and_update_initial_xyz(self, number_of_drones, obstacles_aabbs):
        result = self.initial_xyz
        self.show_initial_xyz_circle()
        if self.randomized_initial_xyz:
            if self.experiment == ExperimentType.E0 or self.experiment == ExperimentType.E1:
                self.initial_xyz = self._reshape_initial_xyz(randomize_initial_positions(obstacles_aabbs), number_of_drones)

            if self.experiment == ExperimentType.E2 or self.experiment == ExperimentType.E3:
                self.initial_xyz = self._reshape_initial_xyz(get_random_point_in_range(-1, 1, -1, -0.5, 0.1, 1), number_of_drones)

        self.previous_xyz = result
        return self._reshape_initial_xyz(result, number_of_drones)

    def _reshape_initial_xyz(self, init_xyz, number_of_drones):
        return (np.vstack(init_xyz)
         .transpose()
         .reshape(number_of_drones, 3))

    def update_target_point(self, obstacles_aabbs):
        if self.randomized_target_point:
            if self.experiment == ExperimentType.E0 or self.experiment == ExperimentType.E1:
                # self.target_point = randomize_target_point(obstacles_aabbs)
                self.target_point = get_opposite_point(self.previous_xyz[0])
            elif self.experiment == ExperimentType.E2 or self.experiment == ExperimentType.E3:
                self.target_point = get_random_point_in_range(-1, 1, 0.5, 1, 0.1, 1)
        self.show_target_point_circle()
        return self.target_point

    def get_number_of_collisions(self):
        return self.number_of_collisions

    def get_and_update_collisions_pos_orn(self):
        result = []
        if self.randomized_collisions:
            if self.experiment == ExperimentType.E0 or self.experiment == ExperimentType.E1:
                for i in range(self.number_of_collisions):
                    pos, orn = get_random_obstacle_pose_between_radii(0, 0.4)
                    result.append((pos, orn))
            elif self.experiment == ExperimentType.E2 or self.experiment == ExperimentType.E3:
                result = generate_field_of_obstacles(5, 2, 0.8, 0.3, 0.5)
        else:
            result = self.collisions_pos_orn
        return result

    def show_target_point_circle(self):
        if self.show_gizmos:
            # Add target point transparent circle
            p.loadURDF(get_norm_path("../assets/target_circle.urdf"),
                       self.target_point,
                       useFixedBase=True)

    def show_local_target_point_circle(self):
        if self.show_gizmos:
            # Add target point transparent circle
            p.loadURDF(get_norm_path("../assets/target_circle.urdf"),
                       self.local_target_point,
                       useFixedBase=True)

    def show_initial_xyz_circle(self):
        if self.show_gizmos:
            # Add initial point transparent circle
            p.loadURDF(get_norm_path("../assets/initial_circle.urdf"),
                       self.initial_xyz[0],
                       useFixedBase=True)

    def get_show_gizmos(self):
        return self.show_gizmos
