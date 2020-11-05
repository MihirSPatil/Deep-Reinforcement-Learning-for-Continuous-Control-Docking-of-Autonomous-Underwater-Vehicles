import numpy as np
from gym import spaces
import random
from deepleng_gym.robot_envs import deepleng_env
from deepleng_gym.log_utils import write_to_json
from gym.envs.registration import register

timestep_limit_per_episode = 1100  # Can be any Value
# timestep_limit_per_episode = int(6e5)  # Can be any Value

register(
    id='DeeplengDocking-v2',
    entry_point='deepleng_gym.task_envs.deepleng.deepleng_docking:DeeplengDockingEnv',
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
        self.num_episodes = 0
        self.data_dict = dict()

        '''Actions and Observations'''
        self.obs_thruster_rpm = 150

        self.obs_linear_vel = 2.0
        self.obs_angular_vel = 1.0
        # self.max_linear_vel = 1.5
        # self.min_linear_vel = -1.5
        # self.max_angular_vel = 0.5
        # self.min_angular_vel = -0.5

        '''Desired position is different set so that only the AUV's nose is in the docking station'''

        self.desired_pose = dict(position=dict(x=0.0, y=0.0), orientation=dict(p=0.0, y=0.0))
        self.docking_station_origin = dict(x=0.0, y=0.0, z=0.0)

        self.desired_thruster_rpm = dict(x=0.0, y_rear=0.0, y_front=0.0)

        '''roll, pitch and yaw are always in radians'''
        # maybe remove roll from the observations since it cannot be controlled anyway

        self.reward_threshold_pitch = 0.8
        self.reward_threshold_yaw = np.pi / 2

        self.obs_threshold_pitch = np.pi
        self.obs_threshold_yaw = np.pi

        '''
        The AUV position needs to be inside this defined limits, i.e it needs to be less than these
        defined limits excluding these limits themselves
        '''
        self.work_space_x_max = 9.0
        self.work_space_x_min = -9.0
        self.work_space_y_max = 9.0
        self.work_space_y_min = -9.0
        self.work_space_z_max = -0.5
        self.work_space_z_min = -9.0

        # We place the Maximum and minimum values of observations
        '''
        Since the docking station is the desired position we set the max and min values around
        it's pose and velocity by adding and subtracting the max and min values of the robot's workspace
        '''
        # observation = [x, y, pitch, yaw, x_r_dot, y_r_dot, z_r_dot, roll_dot, pitch_dot, yaw_dot,
        # thruster1, thruster2, thruster3, thruster4, thruster5]
        high = np.array([self.work_space_x_max,
                         self.work_space_y_max,
                         # self.work_space_z_max,
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

        low = np.array([self.work_space_x_min,
                        self.work_space_y_min,
                        # self.work_space_z_min,
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

        self.observation_space = spaces.Box(low, high)
        np.set_printoptions(precision=5, suppress=True)

        # print("ACTION SPACES TYPE: ", type(self.action_space))
        # print("OBSERVATION SPACES TYPE: ", type(self.observation_space))
        # print("Observation size= ", self.observation_space.shape)

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeeplengDockingEnv, self).__init__()

    def normalize_angle(self, angle):
        if angle <= -np.pi:
            angle += 2 * np.pi
        if angle > np.pi:
            angle -= 2 * np.pi

        return angle

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
                                           (self.work_space_x_max - delta, self.work_space_x_max - delta)])), 5)
        y = round(
            random.uniform(*random.choice([(self.work_space_y_min + delta, self.work_space_y_min + initialization_band),
                                           (self.work_space_y_max - delta, self.work_space_y_max - delta)])), 5)
        # z = round(random.uniform(self.work_space_z_min+delta, self.work_space_z_max - delta),2)

        pitch = round(random.uniform(-self.reward_threshold_pitch, self.reward_threshold_pitch), 5)
        yaw = round(np.arctan2(-y, -x), 5) + np.random.normal(0, 0.2)

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
        self.ep_reward = list()
        self.num_episodes += 1
        init_pose = self.get_auv_pose()
        # print("Init Pose: ", init_pose)
        init_vel = self.get_auv_velocity()
        # print("Init Vel: ", init_vel)
        init_thruster_rpms = self.get_thruster_rpm()
        # print("Init Rpms: ", init_thruster_rpms)

        # print("Init_observation: ", np.round(np.hstack((init_pose, init_vel, init_thruster_rpms)), 2))

    def _set_action(self, action):
        # action comes from the DRL algo we are using
        """
        It sets the thruster rpm of the deepleng auv based on the rpm values given
        by the DRL algo we are using.
        :param action: The action vector containing rpm values for the 5 thrusters.
        """

        assert action.shape == (3,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        # print("Unscaled action in task env: {}".format(action))
        action = np.interp(action,
                           (self.action_space.low[0], self.action_space.high[0]),
                           (-self.obs_thruster_rpm, self.obs_thruster_rpm))
        # print("Scaled action in task env: {}".format(action))

        # action[1] = -150 # these are only to be used when checking the velocity transforms
        # action[0] = 0
        # action[2] = 150

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
        obs_data_pose = self.get_auv_pose()

        obs_data_vel = self.get_auv_velocity()

        obs_data_thruster_rpm = self.get_thruster_rpm()
        obs_data_thruster_rpm = np.interp(obs_data_thruster_rpm,
                                          (-self.obs_thruster_rpm, self.obs_thruster_rpm),
                                          (self.action_space.low[0], self.action_space.high[0]))

        self.obs_data_thrust = self.get_thruster_thrust()

        observation = np.round(np.hstack((obs_data_pose, obs_data_vel, obs_data_thruster_rpm)), 5)
        print("Observation: {}".format(observation))

        # list of 13 elements, 0-3: pose, 4-9: velocity, 10-13: thruster rpms
        return observation

    def _in_docking_cone(self, observations):
        """
        Function to check whether the x and y coordinates of the auv nose lie within a 3d pyramid region
        projecting outwards of the docking station entrance. The height of this pyramid is presently set
        to 3.5 along the -ve x axis
        """
        x, y = observations[0], observations[1]
        auv_in_cone = False
        x_cone = -3.5
        lower_x = x_cone <= x
        upper_x = x <= 0
        upper_y = y <= np.tan(-0.4) * x_cone
        lower_y = y >= np.tan(0.4) * x_cone
        # upper_z = z <= tan(0.3) * x_cone
        # lower_z = z >= tan(0.3) * x_cone
        # print("upper_x: {}, lower_x: {}, upper_y: {}, lower_y: {}".format(upper_x, lower_x, upper_y, lower_y))

        if upper_x and lower_x and upper_y and lower_y:  # and upper_z and lower_z):
            auv_in_cone = True
            # print("Inside cone")
        return auv_in_cone

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
        # print("Outside limit done: ", not (is_within_limit))
        # print("Reached goal: ", has_reached_des_point)
        # print("Done: ", done)

        return done

    def has_reached_goal(self, current_observation):
        """
        It return True if the current position is similar to the desired position
        otherwise return false. Checks only the first 6 elements of the observation
        since these represent the pose.
        """
        # auv_desired_pose = np.reshape(np.array([list(x.values()) for x in self.desired_pose.values()]), -1)
        dock_origin = np.fromiter(self.docking_station_origin.values(), dtype=float)

        is_in_desired_pos = False
        # if (np.round(abs(current_observation[:2]) - abs(dock_origin[:2]), 2) <= 3.5).all():
        if (np.round(abs(current_observation[:2]) - abs(dock_origin[:2]), 2) <= 1).all() and \
                -0.3 <= (current_observation[3] - self.desired_pose['orientation']['p']) <= 0.3 and \
                self._in_docking_cone(current_observation):
            is_in_desired_pos = True
        # print("Has reached goal: {}".format(is_in_desired_pos))
        return is_in_desired_pos

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
        # Reward weights:
        w_euc = 30
        w_x = 0.4
        w_y = 1.2
        w_z = 0
        w_u = 0.5  # surge(forward) velocity in body frame
        w_v = 100  # sway velocity in body frame
        w_w = 50  # heave velocity in body frame
        w_yaw = - 1.3
        wt_x = 2
        wt_y = 5
        wt_z = 5

        auv_desired_pose = np.reshape(np.array([list(x.values()) for x in self.desired_pose.values()]), -1)

        obs_diff = np.round(observations[:4] - auv_desired_pose, 2)  # [x, y, pitch, yaw]

        distance_reward = np.linalg.norm(obs_diff[:2])

        thruster_reward = - wt_x * np.abs(observations[-3]) \
                          - wt_y * np.abs(observations[-2]) \
                          - wt_y * np.abs(observations[-1])  # \
        # - wt_z * np.abs(observations[-2]) \
        # - wt_z * np.abs(observations[-1])

        if self._in_docking_cone(observations):
            align_reward = - w_y * abs(obs_diff[1]) \
                           - w_yaw * abs(obs_diff[-1])  # \
            #  - w_pitch * abs(obs_diff[-2]) \
            #  - w_z * abs(observations[2])
            w_euc = 5

        else:
            align_reward = 0

        #TODO: test with this behind the station penalty
        # if distance_reward <= 3.5 and obs_diff[0] > 0: #maybe also add limit for y
        #   behind_station_penalty = -100
        # else: behind_station_penalty = 0

        continuous_reward = np.sum(((-w_euc * distance_reward), thruster_reward, align_reward))  # , behind_station_penalty))
        # continuous_reward = np.sum((-distance_reward))  # , behind_station_penalty))

        if not done:
            if self.is_inside_workspace(observations) and not self.has_reached_goal(observations):
                reward = continuous_reward

        if done:
            if self.has_reached_goal(observations):
                # print("reached goal: {}".format(observations))
                # reward = 10000 + continuous_reward
                reward = 15000 + continuous_reward

            if not self.is_inside_workspace(observations):
                # reward = -20000
                reward = -25000
        # print("Reward: {}".format(reward))
        self.ep_reward.append(round(reward, 2))

        if done:
            print("Episode_num: {}".format(self.num_episodes))
            print("Ep_Reward: {}".format(self.ep_reward))
            self.data_dict[self.num_episodes] = self.ep_reward
            write_to_json(self.data_dict) #randomly doesn't work with the spinningup off-policy algos

        return round(reward, 2)