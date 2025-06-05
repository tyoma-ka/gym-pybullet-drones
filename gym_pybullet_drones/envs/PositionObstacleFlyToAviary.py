import os

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.BaseRLFlyToAviary import BaseRLFlyToAviary
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ExperimentType
from gym_pybullet_drones.utils.utils import get_norm_path, is_close_to_obstacle, \
    draw_target_circle, is_close_to_obstacle_with_distance, get_obstacle_term, normalize, get_closest_obstacle_distance


class PositionObstacleFlyToAviary(BaseRLFlyToAviary):
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
        roll = state[7]
        pitch = state[8]
        vel = state[10:13]

        # --- Target and distance ---
        target_point = self.training_state_controller.get_target_point()
        vec_to_target = target_point - pos

        squared_dist = np.dot(vec_to_target, vec_to_target)
        r_gauss = np.exp(-squared_dist / 2.0)

        # --- Altitude shaping term ---
        desired_z = target_point[2]
        altitude_error = 1.0 - (pos[2] / desired_z)
        r_altitude = np.exp(-altitude_error ** 2)

        # --- Smoothness term (reward low velocity) ---
        v_mag = np.linalg.norm(vel)
        r_smooth = np.exp(-v_mag ** 2 / 2.0)

        # --- Stability term (reward smaller tilts) ---
        r_stability = np.exp(-(pitch ** 2 + roll ** 2))

        # --- Obstacle term
        closest_distance = get_closest_obstacle_distance(self.raytraced_distances, 0)

        ### Obstacle Approach Penalty term
        # r_cos_obstacle = get_obstacle_term(self.raytraced_directions, vel, 0, self.training_state_controller.laser_range)
        # r_obstacle = 1 - r_cos_obstacle

        ### Continuous term
        # r_obstacle = normalize(closest_distance, 0, self.training_state_controller.laser_range)

        ### Discrete term
        r_obstacle = 1 if normalize(closest_distance, 0, self.training_state_controller.laser_range) == 1 else 0

        self._showVelocityVector(pos, vel)

        if self._droneReachedTargetPoint(0.1):
            r_precision = np.exp(-squared_dist / 0.005)  # Sharper curve near 0
            r_gauss += 2.0 * r_precision
            r_obstacle = 1

        return r_gauss * 1.5 + r_altitude * 1.5 + r_smooth * 0.5 + r_stability * 0.5 + r_obstacle

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
