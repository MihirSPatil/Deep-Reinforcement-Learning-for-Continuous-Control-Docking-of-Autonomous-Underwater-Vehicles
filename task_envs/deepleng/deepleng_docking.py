import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import deepleng_env
from gym.envs.registration import register
from geometry_msgs.msg import Pose, Point
from geometry_msgs.msg import Vector3
from tf.transformations import euler_from_quaternion

timestep_limit_per_episode = 10  # Can be any Value

register(
    id='DeeplengDocking-v0',
    entry_point='openai_ros.task_envs.deepleng.deepleng_docking:DeeplengDockingEnv',
    max_episode_steps=timestep_limit_per_episode,
)


class DeeplengDockingEnv(deepleng_env.DeeplengEnv):
    def __init__(self):
        """
        Make Deepleng learn how to dock to the docking station from a
        starting point within a given range around the docking station.
        """

        # Only variable needed to be set here

        rospy.logdebug("Start DeeplengDockingEnv INIT...")
        # number_actions = rospy.get_param('/wamv/n_actions')
        # number_actions = 4
        self.action_space = spaces.Box(-60, 60, (5,))

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-100, 100)

        # Actions and Observations
        self.max_thruster_rpm = 60
        self.min_thruster_rpm = -60
        self.max_distance_from_des_point = 5
        # self.propeller_high_speed = rospy.get_param('/wamv/propeller_high_speed')
        # self.propeller_low_speed = rospy.get_param('/wamv/propeller_low_speed')
        # self.max_angular_speed = rospy.get_param('/wamv/max_angular_speed')
        # self.max_distance_from_des_point = rospy.get_param('/wamv/max_distance_from_des_point')

        # Get Desired Point to Get
        self.desired_pose = Pose()
        self.desired_pose.position.x = 0.0
        self.desired_pose.position.y = 0.0
        self.desired_pose.position.z = -2.0
        self.desired_pose.orientation.x = 0.0
        self.desired_pose.orientation.y = 0.0
        self.desired_pose.orientation.z = 0.0

        self.desired_point_epsilon = 0.5

        # self.desired_point.x = rospy.get_param("/wamv/desired_point/x")
        # self.desired_point.y = rospy.get_param("/wamv/desired_point/y")
        # self.desired_point.z = rospy.get_param("/wamv/desired_point/z")
        # self.desired_point_epsilon = rospy.get_param("/wamv/desired_point_epsilon")

        self.work_space_x_max = 5
        self.work_space_x_min = -5
        self.work_space_y_max = 5
        self.work_space_y_min = -5
        self.work_space_z_max = 0
        self.work_space_z_min = -5

        self.dec_obs = 1

        # self.work_space_x_max = rospy.get_param("/wamv/work_space/x_max")
        # self.work_space_x_min = rospy.get_param("/wamv/work_space/x_min")
        # self.work_space_y_max = rospy.get_param("/wamv/work_space/y_max")
        # self.work_space_y_min = rospy.get_param("/wamv/work_space/y_min")

        # self.dec_obs = rospy.get_param("/wamv/number_decimals_precision_obs")

        # We place the Maximum and minimum values of observations

        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            3.14,
                            3.14,
                            3.14,
                            self.max_thruster_rpm,
                            self.max_thruster_rpm,
                            self.max_thruster_rpm,
                            self.max_thruster_rpm,
                            self.max_thruster_rpm,
                            self.max_distance_from_des_point
                            ])

        low = numpy.array([self.work_space_x_min,
                           self.work_space_y_min,
                           self.work_space_z_min,
                           -3.14,
                           -3.14,
                           -3.14,
                           self.min_thruster_rpm,
                           self.min_thruster_rpm,
                           self.min_thruster_rpm,
                           self.min_thruster_rpm,
                           self.min_thruster_rpm,
                           0.0
                           ])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE===>" + str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards

        # self.done_reward =rospy.get_param("/wamv/done_reward")
        self.done_reward = 100
        # self.closer_to_point_reward = rospy.get_param("/wamv/closer_to_point_reward")
        self.closer_to_point_reward = 10

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(DeeplengDockingEnv, self).__init__()

        rospy.logdebug("END DeeplengDockingEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the two proppelers speed to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """
        thruster_rpms = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.set_thruster_speed(thruster_rpms, time_sleep=1.0)

        # right_propeller_speed = 0.0
        # left_propeller_speed = 0.0
        # self.set_propellers_speed(  right_propeller_speed,
        #                             left_propeller_speed,
        #                             time_sleep=1.0)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """

        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        auv_pose = self.get_pose()
        current_pose = Pose()
        current_pose.position.x = auv_pose.position.x
        current_pose.position.y = auv_pose.position.y
        current_pose.position.z = auv_pose.position.z
        current_pose.orientation.x = auv_pose.orientation.x
        current_pose.orientation.y = auv_pose.orientation.y
        current_pose.orientation.z = auv_pose.orientation.z
        current_pose.orientation.w = auv_pose.orientation.w

        self.previous_distance_from_des_point = self.get_distance_from_desired_point(current_pose.position)

    # TODO:

    def _set_action(self, action):
        # action comes from the DRL algo we are using
        """
        It sets the joints of wamv based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """

        rospy.logdebug("Start Set Action ==>" + str(action))

        assert action.shape == (5,)
        action = action.copy()  # ensure that we don't change the action outside of this scope

        # Apply action to simulation.
        self.set_thruster_speed(action, time_sleep=1.0)

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        WamvEnv API DOCS.
        :return: observation
        """
        rospy.logdebug("Start Get Observation ==>")

        obs_data = self.get_pose()
        base_position = obs_data.position
        base_orientation_quat = obs_data.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)
        # todo: add the velocities to the observation
        # base_speed_linear = odom.twist.twist.linear
        # base_speed_angular_yaw = odom.twist.twist.angular.z

        distance_from_desired_point = self.get_distance_from_desired_point(base_position)

        observation = []
        observation.append(round(base_position.x, self.dec_obs))
        observation.append(round(base_position.y, self.dec_obs))
        observation.append(round(base_position.z, self.dec_obs))

        observation.append(round(base_roll, self.dec_obs))
        observation.append(round(base_pitch, self.dec_obs))
        observation.append(round(base_yaw, self.dec_obs))

        # todo: add the velocities to the observation
        # observation.append(round(base_speed_linear.x, self.dec_obs))
        # observation.append(round(base_speed_linear.y, self.dec_obs))
        #
        # observation.append(round(base_speed_angular_yaw, self.dec_obs))

        observation.append(round(distance_from_desired_point, self.dec_obs))

        return observation

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The wamvs is ouside the workspace
        2) It got to the desired point
        """
        distance_from_desired_point = observations[-1]

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]

        # todo: add orientation also to check limits
        is_within_limit = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_pose(current_position, self.desired_point_epsilon)

        done = not(is_within_limit) or has_reached_des_point

        return done

    def _compute_reward(self, observations, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """

        # We only consider the plane, the fluctuation in z is due mainly to wave
        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        if not done:

            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE TO GOAL")
                reward = self.closer_to_point_reward
            else:
                rospy.logerr("INCREASE IN DISTANCE TO GOAL")
                reward = -1 * self.closer_to_point_reward

        else:

            if self.is_in_desired_pose(current_position, self.desired_point_epsilon):
                reward = self.done_reward
            else:
                reward = -1 * self.done_reward

        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    # Internal TaskEnv Methods

    def is_in_desired_pose(self, current_position, epsilon=0.05):
        # todo: use orientation also to check the pose
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = self.desired_pose.position.x + epsilon
        x_pos_minus = self.desired_pose.position.x - epsilon
        y_pos_plus = self.desired_pose.position.y + epsilon
        y_pos_minus = self.desired_pose.position.y - epsilon
        z_pos_plus = self.desired_pose.position.z + epsilon
        z_pos_minus = self.desired_pose.position.z - epsilon

        x_current = current_position.x
        y_current = current_position.y
        z_current = current_position.z

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        z_pos_are_close = (z_current <= z_pos_plus) and (z_current > z_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close and z_pos_are_close

        rospy.logdebug("###### IS DESIRED POS ? ######")
        rospy.logdebug("current_position" + str(current_position))
        rospy.logdebug("x_pos_plus" + str(x_pos_plus) + ",x_pos_minus=" + str(x_pos_minus))
        rospy.logdebug("y_pos_plus" + str(y_pos_plus) + ",y_pos_minus=" + str(y_pos_minus))
        rospy.logdebug("z_pos_plus" + str(z_pos_plus) + ",z_pos_minus=" + str(z_pos_minus))
        rospy.logdebug("x_pos_are_close" + str(x_pos_are_close))
        rospy.logdebug("y_pos_are_close" + str(y_pos_are_close))
        rospy.logdebug("z_pos_are_close" + str(z_pos_are_close))
        rospy.logdebug("is_in_desired_pos" + str(is_in_desired_pos))
        rospy.logdebug("############")

        return is_in_desired_pos

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_pose)

        return distance

    def get_distance_from_point(self, current_position, desired_position):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        # print(current_position)
        # print(desired_position)
        a = numpy.array((current_position.x, current_position.y, current_position.z))
        b = numpy.array((desired_position.position.x, desired_position.position.y, desired_position.position.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def is_inside_workspace(self, current_position):
        """
        Check if the Wamv is inside the Workspace defined
        """
        is_inside = False

        rospy.logwarn("##### INSIDE WORK SPACE? #######")
        rospy.logwarn("XYZ current_position" + str(current_position))
        rospy.logwarn(
            "work_space_x_max" + str(self.work_space_x_max) + ",work_space_x_min=" + str(self.work_space_x_min))
        rospy.logwarn(
            "work_space_y_max" + str(self.work_space_y_max) + ",work_space_y_min=" + str(self.work_space_y_min))
        rospy.logwarn("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                is_inside = True

        return is_inside
