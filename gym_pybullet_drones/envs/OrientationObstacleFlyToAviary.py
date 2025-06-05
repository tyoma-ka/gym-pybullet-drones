import os

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.BaseRLFlyToAviary import BaseRLFlyToAviary
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ExperimentType
from gym_pybullet_drones.utils.utils import get_norm_path, is_close_to_obstacle, \
    draw_target_circle, is_close_to_obstacle_with_distance, get_obstacle_term, normalize, get_closest_obstacle_distance, \
    compute_avoidance_point


class OrientationObstacleFlyToAviary(BaseRLFlyToAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 training_state_controller: TrainingStateController = None,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = training_state_controller.episode_length

        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         training_state_controller=training_state_controller
                         )

    ################################################################################

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        # --- Target and distance ---
        target_point = self.training_state_controller.get_target_point()
        vec_to_target = target_point - pos
        squared_dist = np.dot(vec_to_target, vec_to_target)
        distance = np.linalg.norm(vec_to_target)

        # --- Direction term
        if distance > 0 and np.linalg.norm(vel) > 0:
            direction_reward = np.dot(vec_to_target, vel) / (distance * np.linalg.norm(vel))
        else:
            direction_reward = 0.0

        # Only reward for flying toward, not away
        r_direction = max(0, direction_reward)

        closest_distance = get_closest_obstacle_distance(self.raytraced_distances, 0)

        ### Obstacle Approach Penalty term
        r_cos_obstacle = get_obstacle_term(self.raytraced_directions, vel, 0, self.training_state_controller.laser_range)
        # r_obstacle = 1 - r_cos_obstacle

        ### Continuous term
        # r_obstacle = normalize(closest_distance, 0, self.training_state_controller.laser_range)

        ### Discrete term
        r_obstacle = 1 if normalize(closest_distance, 0, self.training_state_controller.laser_range) == 1 else 0

        if closest_distance < self.training_state_controller.laser_range:
            new_local_target = compute_avoidance_point(pos, self.closest_vector_to_obstacle, 0.2)
            self.training_state_controller.set_local_target_point(new_local_target)
        else:
            self.training_state_controller.set_local_target_point(None)

        self._showVelocityVector(pos, vel)

        if self._droneReachedTargetPoint(0.1):
            r_direction = 2
            r_obstacle = 2

        return 2 * r_direction + r_obstacle

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        return self._droneReachedTargetPoint()

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        dist = self.training_state_controller.get_target_point() - self.pos[0, :]
        # dist @ dist > (self.target_point_controller.get_target_point() @ self.target_point_controller.get_target_point()) * 2  # Truncate when the drone is too far away
        #                 or

        if abs(self.rpy[0, 0]) > .4 or abs(self.rpy[0, 1]) > .4:
            return True
        if self._droneHitObstacle():
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False
