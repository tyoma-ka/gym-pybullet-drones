import os

import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.envs.BaseRLFlyToAviary import BaseRLFlyToAviary
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.utils.utils import get_norm_path

import pybullet as p


class HoverAviary(BaseRLFlyToAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
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
        self.EPISODE_LEN_SEC = 8
        self.total_timesteps = 0

        self.target_point_controller = training_state_controller
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

    def _addObstacles(self):
        pass
        # p.loadURDF(get_norm_path("../assets/box_obstacle.urdf"),
        #        [1, 0, .1],
        #        p.getQuaternionFromEuler([0, 0, 0]),
        #        physicsClientId=self.CLIENT
        #        )

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        # --- Target and distance ---
        target_point = self.target_point_controller.get_target_point();
        vec_to_target = target_point - pos
        squared_dist = np.dot(vec_to_target, vec_to_target)
        r_gauss = np.exp(-squared_dist / 2.0) * 2

        # --- Altitude shaping term ---
        desired_z = target_point[2]
        altitude_error = 1.0 - (pos[2] / desired_z)
        r_altitude = np.exp(-altitude_error ** 2)

        # --- Smoothness term (reward low velocity) ---
        r_smooth = np.exp(-np.linalg.norm(vel) ** 2 / 2.0)

        return r_gauss + r_altitude + 0.5 * r_smooth

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        threshold = .0001
        dist = self.target_point_controller.get_target_point() - self.pos[0, :]
        if dist @ dist < threshold * threshold:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        dist = self.target_point_controller.get_target_point() - self.pos[0, :]
        if (dist @ dist > (self.target_point_controller.get_target_point() @ self.target_point_controller.get_target_point()) * 2  # Truncate when the drone is too far away
                or abs(self.rpy[0, 0]) > .4 or abs(self.rpy[0, 1]) > .4  # Truncate when the drone is too tilted
        ):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True

        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
