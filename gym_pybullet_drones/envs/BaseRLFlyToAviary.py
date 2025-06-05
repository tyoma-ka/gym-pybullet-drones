import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.TrainingStateController import TrainingStateController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType, ExperimentType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import is_close_to_obstacle_with_distance, get_norm_path


class BaseRLFlyToAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 training_state_controller: TrainingStateController = None,
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=True,  # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False,  # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         training_state_controller=training_state_controller,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

    ################################################################################

    def _addObstacles(self):
        self.obstacles_aabbs = []
        self.obstacle_ids = []
        pos_orn_list = self.training_state_controller.get_and_update_collisions_pos_orn()
        for pos, orn in pos_orn_list:
            obstacle_id = p.loadURDF(get_norm_path("../assets/box_obstacle.urdf"),
                                     pos,
                                     orn,
                                     physicsClientId=self.CLIENT
                                     )
            self.obstacle_ids.append(obstacle_id)
            # Get AABB (axis-aligned bounding box)
            aabb_min, aabb_max = p.getAABB(obstacle_id, physicsClientId=self.CLIENT)
            self.obstacles_aabbs.append((aabb_min, aabb_max))

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1 * np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1 * np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k, :] = np.array(self.HOVER_RPM * (1 + 0.05 * target))
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                          cur_pos=state[0:3],
                                                          cur_quat=state[3:7],
                                                          cur_vel=state[10:13],
                                                          cur_ang_vel=state[13:16],
                                                          target_pos=next_pos
                                                          )
                rpm[k, :] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                         cur_pos=state[0:3],
                                                         cur_quat=state[3:7],
                                                         cur_vel=state[10:13],
                                                         cur_ang_vel=state[13:16],
                                                         target_pos=state[0:3],  # same as the current position
                                                         target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                         target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector
                                                         # target the desired velocity vector
                                                         )
                rpm[k, :] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k, :] = np.repeat(self.HOVER_RPM * (1 + 0.05 * target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3] + 0.1 * np.array([0, 0, target[0]])
                                                        )
                rpm[k, :] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 15
            #### Observation vector ###   Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ    Dx Dy Dz
            lo = -np.inf
            hi = np.inf
            point_lo = -self.training_state_controller.laser_range
            point_hi = self.training_state_controller.laser_range
            obs_lower_bound = np.array([[lo, lo, lo,  # Orientation (Roll, Pitch, Yaw)
                                         lo, lo, lo,  # Linear velocity (Vx, Vy, Vz)
                                         lo, lo, lo,  # Angular velocity (Wx, Wy, Wz)
                                         lo, lo, lo,  # Vector distance to the target (Tx, Ty, Tz)
                                         -2, -2, -2,
                                         point_lo, point_lo, point_lo]
                                        # Difference between the current velocity vector and the direction to the target
                                        for _ in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi, hi, hi,
                                         hi, hi, hi,
                                         hi, hi, hi,
                                         hi, hi, hi,
                                         2, 2, 2,
                                         point_hi, point_hi, point_hi]
                                        for _ in range(self.NUM_DRONES)])
            #### Add lidar point cloud ######
            # pointcloud_lower_bound = np.zeros((self.NUM_DRONES, self.NUM_RAYS))
            # pointcloud_upper_bound = np.ones(
            #     (self.NUM_DRONES, self.NUM_RAYS)) # * self.training_state_controller.laser_range
            #
            # obs_lower_bound = np.hstack([obs_lower_bound, pointcloud_lower_bound])
            # obs_upper_bound = np.hstack([obs_upper_bound, pointcloud_upper_bound])

            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE == ActionType.PID:
                    obs_lower_bound = np.hstack(
                        [obs_lower_bound, np.array([[act_lo, act_lo, act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack(
                        [obs_upper_bound, np.array([[act_hi, act_hi, act_hi] for i in range(self.NUM_DRONES)])])
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                                          frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_15 = np.zeros((self.NUM_DRONES, 15))

            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)

                vec_to_target = self.training_state_controller.get_target_point() - obs[0:3]
                vel = obs[10:13]

                if np.linalg.norm(vec_to_target) > 0:
                    unit_to_target = vec_to_target / np.linalg.norm(vec_to_target)
                else:
                    unit_to_target = np.zeros(3)

                if np.linalg.norm(vel) > 0:
                    unit_vel = vel / np.linalg.norm(vel)
                else:
                    unit_vel = np.zeros(3)

                direction_diff = unit_to_target - unit_vel

                obs_15[i, :] = np.hstack([
                    obs[7:10],  # Orientation (Roll, Pitch, Yaw)
                    obs[10:13],  # Linear velocity (Vx, Vy, Vz)
                    obs[13:16],  # Angular velocity (Wx, Wy, Wz)
                    vec_to_target.reshape(3, ),  # Vector distance to the target (Dx, Dy, Dz),
                    direction_diff,
                ]).reshape(15, )
            ret = np.array([obs_15[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            #### Add lidar point cloud
            # ret = np.hstack([ret, np.array([self.raytraced_distances[i] / self.training_state_controller.laser_range for i in range(self.NUM_DRONES)])])

            ret = np.hstack([ret, np.array(
                [self.closest_vector_to_obstacle for i in
                 range(self.NUM_DRONES)])])

            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

    def _showVelocityVector(self, pos, vel):
        if not self.training_state_controller.show_velocity_vector:
            return
        scale = 0.5
        vel_endpoint = [pos[0] + vel[0] * scale,
                        pos[1] + vel[1] * scale,
                        pos[2] + vel[2] * scale]

        p.addUserDebugLine(pos, vel_endpoint, lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0.1,
                           physicsClientId=self.CLIENT)

    def _droneHitObstacle(self):
        drone_id = self.DRONE_IDS[0]
        obstacle_ids = self.obstacle_ids

        for obs_id in obstacle_ids:
            contacts = p.getContactPoints(bodyA=drone_id, bodyB=obs_id)
            if contacts and len(contacts) > 0:
                return True
        return False

    def _droneReachedTargetPoint(self, threshold=None):
        if threshold is None:
            if self.training_state_controller.experiment == ExperimentType.E0 or \
                    self.training_state_controller.experiment == ExperimentType.E2:
                threshold = .05
            else:
                threshold = 0
        dist = self.training_state_controller.target_point - self.pos[0, :]
        if dist @ dist < threshold * threshold:
            return True
        else:
            return False

    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            "hit_obstacle": self._droneHitObstacle(),
            "reached_target": self._droneReachedTargetPoint(0.1),
            "distance_to_target": np.linalg.norm(self.training_state_controller.get_target_point() -
                                                 self._getDroneStateVector(0)[0:3]),
            "velocity": np.linalg.norm(self.vel[0]),
            "orientation": np.linalg.norm(self.rpy[0]),
            "distance_to_obstacle": is_close_to_obstacle_with_distance(self.raytraced_distances, 0,
                                                                       self.training_state_controller.laser_range)[1]
        }
