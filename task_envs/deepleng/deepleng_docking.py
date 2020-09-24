# import rospy
import numpy as np
from gym import spaces
import random
from openai_ros.robot_envs import deepleng_env
from gym.envs.registration import register

timestep_limit_per_episode = 1100  # Can be any Value
# timestep_limit_per_episode = int(1e6)  # Can be any Value

register(
    id='DeeplengDocking-v1',
    entry_point='openai_ros.task_envs.deepleng.deepleng_docking:DeeplengDockingEnv',
    max_episode_steps=timestep_limit_per_episode,
)


# todo: reformat the code for better readability, rename functions appropriately

class DeeplengDockingEnv(deepleng_env.DeeplengEnv):
    def __init__(self):
        """
        Make Deepleng learn how to dock to the docking station from a
        starting point within a given range around the docking station.
        """
        # print("Start DeeplengDockingEnv INIT...")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.reward_range = (-np.inf, np.inf)

        '''Actions and Observations'''
        self.obs_thruster_rpm = 150
        # self.max_thruster_rpm = 150.0
        # self.min_thruster_rpm = -150.0

        self.obs_linear_vel = 2.0
        self.obs_angular_vel = 1.0
        # self.max_linear_vel = 1.5
        # self.min_linear_vel = -1.5
        # self.max_angular_vel = 0.5
        # self.min_angular_vel = -0.5

        '''Desired position is different set so that only the AUV's nose is in the docking station'''

        self.desired_pose = dict(position=dict(x=0.0, y=0.0), orientation=dict(p=0.0, y=0.0))
        self.docking_station_origin = dict(x=0.0, y=0.0, z=0.0)

        # self.desired_vel = dict(linear=dict(x=0.0, y=0.0, z=0.0), angular=dict(x=0.0, y=0.0, z=0.0))

        # self.desired_thruster_rpm = dict(x=0.0, y_rear=0.0, y_front=0.0, diving_cell_rear=0.0, diving_cell_front=0.0)
        self.desired_thruster_rpm = dict(x=0.0, y_rear=0.0, y_front=0.0)

        '''roll, pitch and yaw are always in radians'''
        # maybe remove roll from the observations since it cannot be controlled anyway

        self.reward_threshold_pitch = 0.8
        self.reward_threshold_yaw = 3.14 / 2

        self.obs_threshold_pitch = 3.14
        self.obs_threshold_yaw = 3.14

        '''
        The AUV position needs to be inside this defined limits, i.e it needs to be less than these
        defined limits excluding these limits themselves
        '''
        self.work_space_x_max = 6.0
        self.work_space_x_min = -6.0
        self.work_space_y_max = 6.0
        self.work_space_y_min = -6.0
        self.work_space_z_max = -0.5
        self.work_space_z_min = -6.0

        # We place the Maximum and minimum values of observations
        '''
        Since the docking station is the desired position we set the max and min values around
        it's pose and velocity by adding and subtracting the max and min values of the robot's workspace
        '''
        # observation = [x, y, roll, pitch, yaw, x_r_dot, y_r_dot, z_r_dot, roll_dot, pitch_dot, yaw_dot,
        # thruster1, thruster2, thruster3, thruster4, thruster5]
        high = np.array([self.work_space_x_max,
                         self.work_space_y_max,
                         # self.work_space_z_max,
                         # self.max_roll,
                         self.obs_threshold_pitch,
                         self.obs_threshold_yaw,
                         self.obs_linear_vel,
                         self.obs_linear_vel,
                         self.obs_linear_vel,
                         self.obs_angular_vel,
                         self.obs_angular_vel,
                         self.obs_angular_vel,
                         self.obs_thruster_rpm,
                         self.obs_thruster_rpm,
                         self.obs_thruster_rpm])
        # 15])

        low = np.array([self.work_space_x_min,
                        self.work_space_y_min,
                        # self.work_space_z_min,
                        # self.min_roll,
                        -1 * self.obs_threshold_pitch,
                        -1 * self.obs_threshold_yaw,
                        -1 * self.obs_linear_vel,
                        -1 * self.obs_linear_vel,
                        -1 * self.obs_linear_vel,
                        -1 * self.obs_angular_vel,
                        -1 * self.obs_angular_vel,
                        -1 * self.obs_angular_vel,
                        -1 * self.obs_thruster_rpm,
                        -1 * self.obs_thruster_rpm,
                        -1 * self.obs_thruster_rpm])
        # 0])

        self.observation_space = spaces.Box(low, high)
        np.set_printoptions(precision=2, suppress=True)

        # print("ACTION SPACES TYPE: ", type(self.action_space))
        # print("OBSERVATION SPACES TYPE: ", type(self.observation_space))
        # print("Observation size= ", self.observation_space.shape)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeeplengDockingEnv, self).__init__()

    def _set_init_pose(self):
        """
        Sets the initial pose of the AUV to a random value between 0
        and 7 (excluding 7) meters along the x, y and z axis

        delta: offset to prevent spawning the AUV at the exact limits of the workspace
        initialization_band: used to define the lower limit of the AUV spawn area
        """
        initialization_band = 2
        delta = 0.4
        random.seed(None)

        x = round(
            random.uniform(*random.choice([(self.work_space_x_min + delta, self.work_space_x_min + initialization_band),
                                           (self.work_space_x_max - delta, self.work_space_x_max - delta)])), 2)
        y = round(
            random.uniform(*random.choice([(self.work_space_y_min + delta, self.work_space_y_min + initialization_band),
                                           (self.work_space_y_max - delta, self.work_space_y_max - delta)])), 2)
        # z = round(random.uniform(self.work_space_z_min+delta, self.work_space_z_max - delta),2)

        pitch = round(random.uniform(-self.reward_threshold_pitch, self.reward_threshold_pitch), 2)
        yaw = round(np.arctan2(-y, -x), 2)

        self.set_auv_pose(x, y, -2, 0.0, 0.0, yaw, time_sleep=0.5)
        # self.set_auv_pose(x, y, z, roll, pitch, yaw, time_sleep=1)

        return True

    def _set_init_vel(self):
        """
        Sets the thruster rpms to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """
        thruster_rpms = [0.0, 0.0, 0.0]
        self.set_thruster_speed(thruster_rpms, time_sleep=0.5)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        We get the initial pose to measure the distance from the desired point.
        All the data from the simulator is pre-processed into numpy arrays
        """

        init_pose = self.get_auv_pose()
        # print("Init_Pose: ", init_pose)
        init_vel = self.get_auv_velocity()
        init_thruster_rpms = self.get_thruster_rpm()
        # self.prev_distance_to_goal = self.get_distance_from_point(init_pose[:2], np.array([-1.45, 0]))

        # self.previous_difference_to_goal_state = self.get_difference_to_goal_state(init_pose,
        #                                                                            init_vel,
        #                                                                            init_thruster_rpms)
        print("Init_observation: ", np.round(np.hstack((init_pose, init_vel, init_thruster_rpms))))

    def _set_action(self, action):
        # action comes from the DRL algo we are using
        """
        It sets the thruster rpm of the deepleng auv based on the rpm values given
        by the DRL algo we are using.
        :param action: The action vector containing rpm values for the 5 thrusters.
        """

        assert action.shape == (3,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        action = np.interp(action, (action.min(), action.max()), (-150.0, +150.0))
        # print("Setting action in task env: {}".format(action))

        # Apply action to simulation.
        self.set_thruster_rpm(action, time_sleep=0.5)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        Currently it includes only the vehicle pose, we need to check how
        to include the camera data also.
        :return: observation
        """
        # print("Getting observation")

        # todo: could modify to get both the pose and the camera data(in deepleng_env)
        # getting pose, velocity and thrust observations at the same time
        self.obs_data_pose = self.get_auv_pose()

        self.obs_data_vel = self.get_auv_velocity()

        self.obs_data_thruster_rpm = self.get_thruster_rpm()

        self.obs_data_thrust = self.get_thruster_thrust()

        observation = np.round(np.hstack((self.obs_data_pose, self.obs_data_vel, self.obs_data_thruster_rpm)), 2)
        print("Observation: {}".format(observation))

        # list of 13 elements, 0-3: pose, 4-9: velocity, 10-13: thruster rpms
        return observation

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The auv is outside the workspace
        2) It got to the desired point
        """
        # here we use the actual pose of the AUV to check if it is within workspace limits
        is_within_limit = self.is_inside_workspace(observations)
        # print("inside workspace: {} fn: is_done".format(is_within_limit))

        has_reached_des_point = self.has_reached_goal(observations)

        done = not is_within_limit or has_reached_des_point
        print("Outside limit done: ", not (is_within_limit))
        print("Reached goal: ", has_reached_des_point)
        print("Done: ", done)

        return done

    def has_reached_goal(self, current_observation):
        """
        It return True if the current position is similar to the desired position
        otherwise return false. Checks only the first 6 elements of the observation
        since these represent the pose.
        """

        # to change once the auv learns how to reach the docking cone
        # add yaw and pitch check also
        # circle_radius = 3.5
        # auv_desired_pose = np.reshape(np.array([list(x.values()) for x in self.desired_pose.values()]), -1)
        dock_origin = np.fromiter(self.docking_station_origin.values(), dtype=float)

        is_in_desired_pos = False
        if (np.round(abs(current_observation[:2]) - abs(dock_origin[:2]), 2) <= 3.5).all():
            is_in_desired_pos = True

        return is_in_desired_pos

    # def get_distance_from_point(self, current_position, desired_position):
    #     """
    #     Gets the euclidean distance given a pose array
    #     """
    #     print("Task_env::Current_position", current_position)
    #     # print(desired_position)

    #     distance = round(np.linalg.norm(current_position - desired_position), 2)

    #     return distance

    # def get_difference_to_goal_state(self, current_observation):
    #     """
    #     Calculates the difference between current pose, velocity and thruster rpm w.r.t
    #     the desired pose, velocity and thruster rpm and returns it as a single column vector
    #     """
    #     auv_desired_vel = np.reshape(np.array([list(x.values()) for x in self.desired_vel.values()]), -1)
    #     auv_desired_thruster_rpm = np.array(list(self.desired_thruster_rpm.values()))
    #
    #     pose_diff = auv_desired_pose - current_pose
    #     vel_diff = auv_desired_vel - current_vel
    #     thruster_diff = auv_desired_thruster_rpm - current_thruster_rpms
    #     return np.hstack((pose_diff, vel_diff, thruster_diff))

    def is_inside_workspace(self, current_position):
        """
        Check if the deepleng is inside the defined workspace limits,
        returns true if it is inside the workspace, and false otherwise
        """
        is_inside = False

        # print("Current position: {}".format(current_position))
        if self.work_space_x_min <= current_position[0] <= self.work_space_x_max:
            # print("Within x limits")
            if self.work_space_y_min <= current_position[1] <= self.work_space_y_max:
                # print("Within y limits")
                # if self.work_space_z_min <= current_position[2] <= self.work_space_z_max:
                # print("Within z limits")
                # if self.min_pitch <= current_position[2] <= self.max_pitch:
                #     # print("Within pitch limits")
                #     if self.min_yaw <= current_position[3] <= self.max_yaw:
                        # print("Within yaw limits")
                is_inside = True

        # print("is_inside: ", is_inside)
        return is_inside

    def _compute_reward(self, observations, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """
        # Reward approach:
        w_x = 50
        w_y = 25
        w_z = 50
        w_pitch = 100
        w_u = 1000  # surge(forward) velocity in body frame
        w_v = 500  # sway velocity in body frame
        w_w = 500  # heave velocity in body frame
        limit_u = 0.01

        auv_desired_pose = np.reshape(np.array([list(x.values()) for x in self.desired_pose.values()]), -1)
        obs_diff = np.round(observations[:4] - auv_desired_pose, 2)

        continuous_reward = - (w_x * (obs_diff[0] ** 2)) - (w_y * (obs_diff[1] ** 2)) \
                            - (w_pitch * (obs_diff[2] ** 2)) \
                            - (w_u * (observations[4] ** 2 / max(round(np.linalg.norm(obs_diff[:2] ** 2), 2), limit_u))) \
                            - (w_v * observations[5] ** 2) + (w_u * observations[4])

        if not done:
            if self.is_inside_workspace(observations) and not self.has_reached_goal(observations):
                reward = continuous_reward

        if done:
            if self.has_reached_goal(observations):
                reward = 20000 + continuous_reward

            if not self.is_inside_workspace(observations):
                reward = -20000

        print("Reward: {}".format(round(reward),2))
        return round(reward, 2)
