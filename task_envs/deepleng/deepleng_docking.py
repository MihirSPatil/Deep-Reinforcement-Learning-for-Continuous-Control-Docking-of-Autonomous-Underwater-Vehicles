# import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import deepleng_env
from gym.envs.registration import register

timestep_limit_per_episode = 10000  # Can be any Value

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
        self.action_space = spaces.Box(-150.0, 150.0, (5,))

        # We set the reward range, which is not compulsory. did not set it as I wasn't sure what the range would be
        # self.reward_range = (-20000.0, 20000.0)
        self.reward_range = (-np.inf, np.inf)

        # Actions and Observations
        self.max_thruster_rpm = 150.0
        self.min_thruster_rpm = -150.0

        self.max_linear_vel = 1.5
        self.min_linear_vel = -1.5
        self.max_angular_vel = 0.5
        self.min_angular_vel = -0.5

        # Get Desired Point to Get
        # Desired position is different set so that only the AUV's nose is in the docking station

        self.desired_pose = dict(position=dict(x=-1.45, y=0.0, z=-2.0), orientation=dict(r=0.0, p=0.0, y=0.0))

        self.desired_vel = dict(linear=dict(x=0.5, y=0.0, z=0.0), angular=dict(x=0.0, y=0.0, z=0.0))

        self.desired_thruster_rpm = dict(x=0.0, y_rear=0.0, y_front=0.0, diving_cell_rear=0.0, diving_cell_front=0.0)

        self.desired_point_epsilon = 0.05

        # roll, pitch and yaw are always in radians
        # maybe remove roll from the observations since it cannot be controlled anyway
        self.min_roll = -0.2
        self.min_pitch = -0.87
        self.min_yaw = -3.14 / 2

        # change the values for the roll also
        self.max_roll = 0.2
        self.max_pitch = 0.87
        self.max_yaw = 3.14 / 2

        '''
        The AUV position needs to be inside this defined limits, i.e it needs to be less than these
        defined limits excluding these limits themselves
        '''
        self.work_space_x_max = 5.0
        self.work_space_x_min = -5.0
        self.work_space_y_max = 5.0
        self.work_space_y_min = -5.0
        self.work_space_z_max = -0.5
        self.work_space_z_min = -5.0

        self.dec_obs = 2

        # We place the Maximum and minimum values of observations
        '''
        Since the docking station is the desired position we set the max and min values around
        it's pose and velocity by adding and subtracting the max and min values of the robot's workspace
        '''
        # observation = [x_r, y_r, z_r, roll, pitch, yaw, x_r_dot, y_r_dot, z_r_dot, roll_dot, pitch_dot, yaw_dot,
        # thruster1, thruster2, thruster3, thruster4, thruster5]
        low = np.array([self.desired_pose['position']['x'] - self.work_space_x_max,
                        self.desired_pose['position']['y'] - self.work_space_y_max,
                        self.desired_pose['position']['z'] - self.work_space_z_max,
                        self.desired_pose['orientation']['r'] - self.max_roll,
                        self.desired_pose['orientation']['p'] - self.max_pitch,
                        self.desired_pose['orientation']['y'] - self.max_yaw,
                        self.desired_vel['linear']['x'] - self.max_linear_vel,
                        self.desired_vel['linear']['y'] - self.max_linear_vel,
                        self.desired_vel['linear']['z'] - self.max_linear_vel,
                        self.desired_vel['angular']['x'] - self.max_angular_vel,
                        self.desired_vel['angular']['y'] - self.max_angular_vel,
                        self.desired_vel['angular']['z'] - self.max_angular_vel,
                        self.desired_thruster_rpm['x'] - self.max_thruster_rpm,
                        self.desired_thruster_rpm['y_rear'] - self.max_thruster_rpm,
                        self.desired_thruster_rpm['y_front'] - self.max_thruster_rpm,
                        self.desired_thruster_rpm['diving_cell_rear'] - self.max_thruster_rpm,
                        self.desired_thruster_rpm['diving_cell_front'] - self.max_thruster_rpm])

        high = np.array([self.desired_pose['position']['x'] - self.work_space_x_min,
                         self.desired_pose['position']['y'] - self.work_space_y_min,
                         self.desired_pose['position']['z'] - self.work_space_z_min,
                         self.desired_pose['orientation']['r'] - self.min_roll,
                         self.desired_pose['orientation']['p'] - self.min_pitch,
                         self.desired_pose['orientation']['y'] - self.min_yaw,
                         self.desired_vel['linear']['x'] - self.min_linear_vel,
                         self.desired_vel['linear']['y'] - self.min_linear_vel,
                         self.desired_vel['linear']['z'] - self.min_linear_vel,
                         self.desired_vel['angular']['x'] - self.min_angular_vel,
                         self.desired_vel['angular']['y'] - self.min_angular_vel,
                         self.desired_vel['angular']['z'] - self.min_angular_vel,
                         self.desired_thruster_rpm['x'] - self.max_thruster_rpm,
                         self.desired_thruster_rpm['y_rear'] - self.min_thruster_rpm,
                         self.desired_thruster_rpm['y_front'] - self.min_thruster_rpm,
                         self.desired_thruster_rpm['diving_cell_rear'] - self.min_thruster_rpm,
                         self.desired_thruster_rpm['diving_cell_front'] - self.min_thruster_rpm])

        # todo: how to add the image data to the already existing observations ?
        self.observation_space = spaces.Box(low, high)

        # print("ACTION SPACES TYPE: ", type(self.action_space))
        # print("OBSERVATION SPACES TYPE: ", type(self.observation_space))
        # print("Observation size= ", self.observation_space.shape)

        # Rewards

        # self.done_reward =rospy.get_param("/wamv/done_reward")

        # self.closer_to_point_reward = rospy.get_param("/wamv/closer_to_point_reward")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeeplengDockingEnv, self).__init__()

        # print("END DeeplengDockingEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the initial pose of the AUV to a random value between 0
        and 5 (excluding 5) meters along the x, y and z axis
        """
        # delta to set the limits from -4.9 to 4.9
        delta = 0.1
        roll = 0.0
        x, y = np.round(np.random.uniform(self.work_space_x_min + delta, self.work_space_x_max - delta, 2), 2)
        yaw = np.round(np.random.uniform(self.min_yaw + delta, self.max_yaw - delta), 2)
        pitch = np.round(np.random.uniform(self.min_pitch + delta, self.max_pitch - delta), 2)
        z = np.round(np.random.uniform(self.work_space_z_min + delta, self.work_space_z_max + delta), 2)

        print("Initial pose:{}".format([x, y, z, roll, pitch, yaw]))
        self.set_auv_pose(x, y, z, roll, pitch, yaw, time_sleep=1.5)

        return True

    def _set_init_vel(self):
        """
        Sets the thruster rpms to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """
        thruster_rpms = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.set_thruster_speed(thruster_rpms, time_sleep=1.5)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """

        # For Info Purposes

        self.cumulative_reward = 0.0
        # We get the initial pose to measure the distance from the desired point.
        # these are numpy arrays
        init_pose = self.get_auv_pose()
        init_vel = self.get_auv_velocity()
        init_thruster_rpms = self.get_thruster_rpm()
        self.previous_difference_to_goal_state = self.get_difference_to_goal_state(init_pose,
                                                                                   init_vel,
                                                                                   init_thruster_rpms)

    def _set_action(self, action):
        # action comes from the DRL algo we are using
        """
        It sets the thruster rpm of the deepleng auv based on the rpm values given
        by the DRL algo we are using.
        :param action: The action vector containing rpm values for the 5 thrusters.
        """

        assert action.shape == (5,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        # print("Setting action in task env: {}".format(action))

        # Apply action to simulation.
        self.set_thruster_speed(action, time_sleep=1.5)

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

        difference_to_goal_state = self.get_difference_to_goal_state(self.obs_data_pose,
                                                                     self.obs_data_vel,
                                                                     self.obs_data_thruster_rpm)

        # print(difference_to_goal_state.tolist())

        observation = np.round(difference_to_goal_state, self.dec_obs).tolist()

        # print(observation)

        # list of 16 elements, 0-5: pose, 6-11: velocity, 12-16: thruster rpms
        return observation

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The auv is outside the workspace
        2) It got to the desired point
        """
        # here we use the actual pose of the AUV to check if it is within workspace limits
        is_within_limit = self.is_inside_workspace(self.obs_data_pose)

        has_reached_des_point = self.is_in_desired_pose(observations)

        done = not is_within_limit or has_reached_des_point

        return done

    def _compute_reward(self, observations, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """
        w_x = 0.4
        w_y = 1.2  # more weightage for displacement in y
        w_z = 1.2  # more weightage for displacement in z
        w_pitch = 0.2  # less weightage since pitch of the vehicle is more limited
        w_yaw = 0.4  # more weightage for yaw since the vehicle has more freedom in it's yaw
        # todo: implement a function to calculate the motor torque
        w_motor_torque = 0.05
        w_x_dot = 10000
        w_y_dot = 1000
        w_z_dot = 1000
        w_abs_vel = 0.5  # w_u
        w_pitch_dot = 500
        w_yaw_dot = 500
        penalty = 20000
        terminal_reward = 10000
        l_abs_vel = 0.05  # delta to avoid division by 0
        abs_vel = np.linalg.norm(observations[6:9])
        l_x_dot = 0.01  # delta to avoid division by 0
        l_y_dot = 0.01  # delta to avoid division by 0
        l_z_dot = 0.01  # delta to avoid division by 0
        l_pitch_dot = 0.01  # delta to avoid division by 0
        l_yaw_dot = 0.01  # delta to avoid division by 0
        motor_torque_vec = self.obs_data_thrust
        torque_reward = w_motor_torque * motor_torque_vec**2
        continuous_reward = - (w_x * (observations[0] ** 2)) - (w_y * (observations[1] ** 2)) \
                            - (w_z * (observations[2] ** 2)) - (w_pitch * (observations[4] ** 2)) \
                            - (w_yaw * (observations[5] ** 2)) \
                            - (w_abs_vel * (abs_vel / max(sum([x ** 2 for x in observations[:3]]), l_abs_vel))) \
                            - torque_reward[0] - torque_reward[1] - torque_reward[2] - torque_reward[3] \
                            - torque_reward[4]

        if not done:
            '''
            is not done only if the robot is inside the workspace, but not yet reached the goal
            '''

            if not self.is_in_desired_pose(observations) and self.is_inside_workspace(self.obs_data_pose):
                reward = continuous_reward

            # if self.is_in_desired_pose(observations):  # docking is achieved to desired level of accuracy
            #
            #     reward = continuous_reward + terminal_reward + (w_x_dot / max(observations[6] ** 2, l_x_dot)) \
            #              + (w_y_dot / max(observations[7] ** 2, l_y_dot)) \
            #              + (w_z_dot / max(observations[8] ** 2, l_z_dot)) \
            #              + (w_pitch_dot / max(observations[10] ** 2, l_pitch_dot)) \
            #              + (w_yaw_dot / max(observations[11] ** 2, l_yaw_dot))

            # if not self.is_inside_workspace(self.obs_data_pose):  # auv has gone out of bounds
            #     reward = - penalty

        else:

            if self.is_in_desired_pose(observations):
                reward = continuous_reward + terminal_reward + (w_x_dot / max(observations[6] ** 2, l_x_dot)) \
                         + (w_y_dot / max(observations[7] ** 2, l_y_dot)) \
                         + (w_z_dot / max(observations[8] ** 2, l_z_dot)) \
                         + (w_pitch_dot / max(observations[10] ** 2, l_pitch_dot)) \
                         + (w_yaw_dot / max(observations[11] ** 2, l_yaw_dot))
            else:
                # since the robot will otherwise be outside the workspace limits
                reward = - penalty

        # print("Reward= ", reward)
        self.cumulative_reward += reward
        # print("Cumulative_reward= ", self.cumulative_reward)
        self.cumulated_steps += 1
        # print("Cumulated_steps= ", self.cumulated_steps)

        return reward

    # Internal TaskEnv Methods
    # todo: maybe rename to is_in_desired_state
    def is_in_desired_pose(self, current_observation):
        """
        It return True if the current position is similar to the desired position
        otherwise return false
        """
        is_in_desired_pos = np.allclose(current_observation[:6], 0.0, atol=self.desired_point_epsilon)

        return is_in_desired_pos

    def get_difference_to_goal_state(self, current_pose, current_vel, current_thruster_rpms):
        """
        Calculates the difference between current pose, velocity and thruster rpm w.r.t
        the desired pose, velocity and thruster rpm and returns it as a single column vector
        """
        auv_desired_pose = np.reshape(np.array([list(x.values()) for x in self.desired_pose.values()]), -1)
        auv_desired_vel = np.reshape(np.array([list(x.values()) for x in self.desired_vel.values()]), -1)
        auv_desired_thruster_rpm = np.array(list(self.desired_thruster_rpm.values()))
        pose_diff = auv_desired_pose - current_pose
        vel_diff = auv_desired_vel - current_vel
        thruster_diff = auv_desired_thruster_rpm - current_thruster_rpms
        return np.hstack((pose_diff, vel_diff, thruster_diff))

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
                if self.work_space_z_min <= current_position[2] <= self.work_space_z_max:
                    # print("Within z limits")
                    if self.min_pitch <= current_position[4] <= self.max_pitch:
                        # print("Within pitch limits")
                        if self.min_yaw <= current_position[5] <= self.max_yaw:
                            # print("Within yaw limits")
                            is_inside = True

        # print("is_inside: ", is_inside)
        return is_inside
