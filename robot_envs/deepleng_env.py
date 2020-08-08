import numpy as np
import rospy
import time
import message_filters
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from openai_ros import robot_gazebo_env
from gazebo_msgs.msg import ModelStates, ModelState
from sensor_msgs.msg import JointState
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class DeeplengEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        """
        Initializes a new DeeplengEnv environment.

        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.

        The Sensors: The sensors accesible are the ones considered useful for AI learning.

        Sensor Topic List:
        * /deepleng/deepleng/camera/camera_image: Camera feed from the on-board camera
        * /gazebo/model_states: contains the current pose and velocity of the auv
        * /deepleng/joint_states: contains the thruster rpms

        Actuators Topic List:
        * /deepleng/thrusters/0/input: publish speed of the back thruster
        * /deepleng/thrusters/1/input: publish speed of the rear y-thruster
        * /deepleng/thrusters/2/input: publish speed of the front y-thruster
        * /deepleng/thrusters/3/input: publish speed of the front diving cell
        * /deepleng/thrusters/4/input: publish speed of the rear diving cell


        Args:
        """
        # rospy.logdebug("Starting DeeplengEnv INIT...")

        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(DeeplengEnv, self).__init__(controllers_list=self.controllers_list,
                                          robot_name_space=self.robot_name_space,
                                          reset_controls=False,
                                          start_init_physics_parameters=False,
                                          reset_world_or_sim="SIM")
        '''setting this to "World instead of "SIM" causes the init pose of the AUV to 
        get overwritten by it's spawning pose hence we then cannot set random initial position
        of the AUV'''

        # print("DeeplengEnv unpause...")
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._pose_callback)
        rospy.Subscriber("/deepleng/joint_states", JointState, self._thruster_rpm_callback)

        self.thrust_sub1 = message_filters.Subscriber("/deepleng/thrusters/0/thrust", FloatStamped)
        self.thrust_sub2 = message_filters.Subscriber("/deepleng/thrusters/1/thrust", FloatStamped)
        self.thrust_sub3 = message_filters.Subscriber("/deepleng/thrusters/2/thrust", FloatStamped)
        self.thrust_sub4 = message_filters.Subscriber("/deepleng/thrusters/3/thrust", FloatStamped)
        self.thrust_sub5 = message_filters.Subscriber("/deepleng/thrusters/4/thrust", FloatStamped)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.thrust_sub1,
                                                               self.thrust_sub2,
                                                               self.thrust_sub3,
                                                               self.thrust_sub4,
                                                               self.thrust_sub5], 1, 5)

        self.ts.registerCallback(self._thruster_thrust_callback)
        # self.camera_sub = rospy.Subscriber('/deepleng/deepleng/camera/camera_image', Image, self._image_callback)

        self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input', FloatStamped, queue_size=1)
        self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input', FloatStamped, queue_size=1)
        self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input', FloatStamped, queue_size=1)
        self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input', FloatStamped, queue_size=1)
        self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input', FloatStamped, queue_size=1)
        self.set_deepleng_pose = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        # Checking that the publishers are connected to their respective topics
        self._check_pub_connection()

        self.bridge = CvBridge()

        self.gazebo.pauseSim()

        # rospy.logdebug("Finished DeeplengEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("DeeplengEnv checking all systems ready")
        self._check_all_sensors_ready()
        rospy.logdebug("DeeplengEnv all systems ready")
        return True

    # DeeplengEnv virtual methods
    # ----------------------------
    def _check_all_sensors_ready(self):
        rospy.logdebug("Checking sensors ")
        self._check_pose_ready()
        # self._check_camera_ready()
        rospy.logdebug("Sensors ready")

    def _check_pose_ready(self):
        self.auv_pose = None
        rospy.logdebug("Waiting for /gazebo/model_states to be ready")
        while self.auv_pose is None and not rospy.is_shutdown():
            try:
                self.auv_pose = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
                rospy.logdebug("/gazebo/model_states ready")

            except:
                rospy.logerr("/gazebo/model_states not ready yet, retrying for getting auv pose")
        return self.auv_pose

    # def _check_camera_ready(self):
    #     self.image_data = None
    #     rospy.logdebug("Waiting for /deepleng/deepleng/camera/camera_image ready")
    #     while self.image_data is None and not rospy.is_shutdown():
    #         try:
    #             self.image_data = rospy.wait_for_message("/deepleng/deepleng/camera/camera_image", Image, timeout=1.0)
    #
    #             rospy.logdebug("/deepleng/deepleng/camera/camera_image is ready")
    #
    #         except:
    #             rospy.logerr("/deepleng/deepleng/camera/camera_image not ready yet, retrying for getting image data")
    #     return self.image_data

    def _pose_callback(self, data):
        """
                 geometry_msgs/Pose pose
                      geometry_msgs/Point position
                        float64 x
                        float64 y
                        float64 z
                      geometry_msgs/Quaternion orientation
                        float64 x
                        float64 y
                        float64 z
                        float64 w
        """

        # has the position and orientation of the Deepleng AUV
        auv_pose = data.pose[-1]
        roll, pitch, yaw = euler_from_quaternion([auv_pose.orientation.x,
                                                  auv_pose.orientation.y,
                                                  auv_pose.orientation.z,
                                                  auv_pose.orientation.w])
        self.auv_pose = np.round(np.array([auv_pose.position.x,
                                           auv_pose.position.y,
                                           auv_pose.position.z,
                                           roll,
                                           pitch,
                                           yaw]), 2)

        # self.goal_pose = data.pose[1]
        auv_vel = data.twist[-1]
        self.auv_vel = np.array([auv_vel.linear.x,
                                 auv_vel.linear.y,
                                 auv_vel.linear.z,
                                 auv_vel.angular.x,
                                 auv_vel.angular.y,
                                 auv_vel.angular.z])

    def _thruster_rpm_callback(self, data):
        """The thruster rpms come in the following order:
            [diving_cell_front_joint,
            diving_cell_rear_joint,
            x_thruster_joint,
            y_thruster_front_joint,
            y_thruster_rear_joint]
        """

        thruster_rpms = data.velocity[2:]
        # print("Joint names: {}".format(data.name))
        # print("Thruster callback: {}".format(thruster_rpms))

        '''Setting the thruster rpms to the following order:
            [x_thruster_joint,
            y_thruster_rear_joint,
            y_thruster_front_joint,
            diving_cell_rear_joint,
            diving_cell_front_joint]
        '''
        self.thruster_rpm = np.array([thruster_rpms[2],
                                      thruster_rpms[4],
                                      thruster_rpms[3],
                                      thruster_rpms[1],
                                      thruster_rpms[0]])

    def _thruster_thrust_callback(self, thruster0, thruster1, thruster2, thruster3, thruster4):
        """
        Aggregating the thrust values from all thrusters into a single array
        """
        self.thrust_vector = np.array([thruster0.data,
                                       thruster1.data,
                                       thruster2.data,
                                       thruster3.data,
                                       thruster4.data])

        # print("thrust vector: ", thrust_vector)

    def _check_pub_connection(self):

        rate = rospy.Rate(10)  # 10hz
        while self.x_thruster.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to x_thruster yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("x_thruster Publisher Connected")

        while self.y_thruster_rear.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to y_thruster_rear yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("y_thruster_rear Publisher Connected")

        while self.y_thruster_front.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to y_thruster_front yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("y_thruster_front Publisher Connected")

        while self.diving_cell_rear.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to diving_cell_rear yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("diving_cell_rear Publisher Connected")

        while self.diving_cell_front.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to diving_cell_front yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("diving_cell_front Publisher Connected")

        while self.set_deepleng_pose.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to set_deepleng_pose yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("set_deepleng_pose Publisher Connected")

        rospy.logdebug("All Publishers READY")

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_vel(self):
        """Sets the Robot's initial velocity
        """
        raise NotImplementedError()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------

    def set_auv_pose(self, x, y, z, roll, pitch, yaw, time_sleep):
        """
         It will set the initial pose the deepleng.
        """
        pose_msg = ModelState()
        pose_msg.model_name = "deepleng"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        orient_x, orient_y, orient_z, orient_w = quaternion_from_euler(roll, pitch, yaw)
        pose_msg.pose.orientation.x = orient_x
        pose_msg.pose.orientation.y = orient_y
        pose_msg.pose.orientation.z = orient_z
        pose_msg.pose.orientation.w = orient_w

        # print("Setting auv pose to {}".format((x, y, z)))
        self.set_deepleng_pose.publish(pose_msg)
        self.wait_time_for_execute_movement(time_sleep)

    def set_thruster_speed(self, thruster_rpms, time_sleep):
        """
        It will set the speed of each thruster of the deepleng.
        """

        x_thruster_rpm = FloatStamped()
        y_thruster_rear_rpm = FloatStamped()
        y_thruster_front_rpm = FloatStamped()
        diving_cell_rear_rpm = FloatStamped()
        diving_cell_front_rpm = FloatStamped()

        x_thruster_rpm.data = thruster_rpms[0]
        y_thruster_rear_rpm.data = thruster_rpms[1]
        y_thruster_front_rpm.data = thruster_rpms[2]
        diving_cell_rear_rpm.data = thruster_rpms[3]
        diving_cell_front_rpm.data = thruster_rpms[4]

        self.x_thruster.publish(x_thruster_rpm)
        self.y_thruster_rear.publish(y_thruster_rear_rpm)
        self.y_thruster_front.publish(y_thruster_front_rpm)
        self.diving_cell_front.publish(diving_cell_front_rpm)
        self.diving_cell_rear.publish(diving_cell_rear_rpm)

        self.wait_time_for_execute_movement(time_sleep)

    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(time_sleep)

    def get_auv_pose(self):
        # returns the auv_pose
        # print("Got auv pose: {}".format(self.auv_pose))
        return self.auv_pose

    def get_auv_velocity(self):
        # returns the linear and angular auv_velocity
        return self.auv_vel

    def get_thruster_rpm(self):
        # returns the thrusters rpm
        return self.thruster_rpm

    def get_thruster_thrust(self):
        # returns the thrust generated from the thruster
        return self.thrust_vector
# import numpy as np
# import rospy
# import time
# from openai_ros import robot_gazebo_env
# from gazebo_msgs.msg import ModelStates
# from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
#
#
#
# class DeeplengEnv(robot_gazebo_env.RobotGazeboEnv):
#
#     def __init__(self):
#         """
#         Initializes a new DeeplengEnv environment.
#
#         To check any topic we need to have the simulations running, we need to do two things:
#         1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
#         that are pause for whatever the reason
#         2) If the simulation was running already for some reason, we need to reset the controlers.
#         This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
#         and need to be reseted to work properly.
#
#         The Sensors: The sensors accesible are the ones considered usefull for AI learning.
#
#         Sensor Topic List:
#         * /deepleng/deepleng/camera/camera_image: Camera feed from the on-board camera
#         * /gazebo/model_states: contains the current pose and velocity of the auv
#
#         Actuators Topic List:
#         * /deepleng/thrusters/0/input: publish speed of the back thruster
#         * /deepleng/thrusters/1/input: publish speed of the rear y-thruster
#         * /deepleng/thrusters/2/input: publish speed of the front y-thruster
#         * /deepleng/thrusters/3/input: publish speed of the front diving cell
#         * /deepleng/thrusters/4/input: publish speed of the rear diving cell
#
#
#         Args:
#         """
#         rospy.logdebug("Start DeeplengEnv INIT...")
#         # Variables that we give through the constructor.
#         # None in this case
#
#         # Internal Vars
#         # Doesnt have any accesibles
#         self.controllers_list = []
#
#         # It doesnt use namespace
#         self.robot_name_space = ""
#
#         # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
#         super(DeeplengEnv, self).__init__(controllers_list=self.controllers_list,
#                                             robot_name_space=self.robot_name_space,
#                                             reset_controls=False,
#                                             start_init_physics_parameters=False,
#                                             reset_world_or_sim="SIM")
#
#
#
#         rospy.logdebug("DeeplengEnv unpause1...")
#         self.gazebo.unpauseSim()
#         #self.controllers_object.reset_controllers()
#
#         self._check_all_systems_ready()
#
#
#         # We Start all the ROS related Subscribers and publishers
#         rospy.Subscriber("/gazebo/model_states", ModelStates, self._pose_callback)
#
#         self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input', FloatStamped, queue_size=1)
#         self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input', FloatStamped, queue_size=1)
#         self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input', FloatStamped, queue_size=1)
#         self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input', FloatStamped, queue_size=1)
#         self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input', FloatStamped, queue_size=1)
#
#         # self.publishers_array = []
#         # self._cmd_drive_pub = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)
#
#         # self.publishers_array.append(self.x_thruster)
#         # self.publishers_array.append(self.y_thruster_rear)
#         # self.publishers_array.append(self.y_thruster_front)
#         # self.publishers_array.append(self.diving_cell_front)
#         # self.publishers_array.append(self.diving_cell_rear)
#
#         # self._check_all_publishers_ready()
#
#         self._check_pub_connection()
#
#         self.gazebo.pauseSim()
#
#         rospy.logdebug("Finished DeeplengEnv INIT...")
#
#     # Methods needed by the RobotGazeboEnv
#     # ----------------------------
#
#
#     def _check_all_systems_ready(self):
#         """
#         Checks that all the sensors, publishers and other simulation systems are
#         operational.
#         """
#         rospy.logdebug("DeeplengEnv check_all_systems_ready...")
#         self._check_all_sensors_ready()
#         rospy.logdebug("END DeeplengEnv _check_all_systems_ready...")
#         return True
#
#
#     # CubeSingleDiskEnv virtual methods
#     # ----------------------------
#
#     def _check_all_sensors_ready(self):
#         rospy.logdebug("START ALL SENSORS READY")
#         self._check_pose_ready()
#         rospy.logdebug("ALL SENSORS READY")
#
#
#     def _check_pose_ready(self):
#         self.auv_pose = None
#         rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
#         while self.auv_pose is None and not rospy.is_shutdown():
#             try:
#                 self.auv_pose = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
#                 rospy.logdebug("Current /gazebo/model_states READY=>")
#
#             except:
#                 rospy.logerr("Current /gazebo/model_states not ready yet, retrying for getting auv pose")
#         return self.auv_pose
#
#
#
#     def _pose_callback(self, data):
#         self.auv_pose = data.pose[-1]
#         # names of the models
#         # print("models: {}".format(data.name[0]))
#
#         # poses of each of the models
#         # print("poses: {}".format(data.pose[0]))
#
#     # def _check_all_publishers_ready(self):
#     #     """
#     #     Checks that all the publishers are working
#     #     :return:
#     #     """
#     #     rospy.logdebug("START ALL SENSORS READY")
#     #     for publisher_object in self.publishers_array:
#     #         self._check_pub_connection(publisher_object)
#     #     rospy.logdebug("ALL SENSORS READY")
#
#     def _check_pub_connection(self):
#
#         rate = rospy.Rate(10)  # 10hz
#         while self.x_thruster.get_num_connections() == 0 and not rospy.is_shutdown():
#             rospy.logdebug("No susbribers to x_thruster yet so we wait and try again")
#             try:
#                 rate.sleep()
#             except rospy.ROSInterruptException:
#                 # This is to avoid error when world is rested, time when backwards.
#                 pass
#         rospy.logdebug("x_thruster Publisher Connected")
#
#         while self.y_thruster_rear.get_num_connections() == 0 and not rospy.is_shutdown():
#             rospy.logdebug("No susbribers to y_thruster_rear yet so we wait and try again")
#             try:
#                 rate.sleep()
#             except rospy.ROSInterruptException:
#                 # This is to avoid error when world is rested, time when backwards.
#                 pass
#         rospy.logdebug("y_thruster_rear Publisher Connected")
#
#         while self.y_thruster_front.get_num_connections() == 0 and not rospy.is_shutdown():
#             rospy.logdebug("No susbribers to y_thruster_front yet so we wait and try again")
#             try:
#                 rate.sleep()
#             except rospy.ROSInterruptException:
#                 # This is to avoid error when world is rested, time when backwards.
#                 pass
#         rospy.logdebug("y_thruster_front Publisher Connected")
#
#         while self.diving_cell_rear.get_num_connections() == 0 and not rospy.is_shutdown():
#             rospy.logdebug("No susbribers to diving_cell_rear yet so we wait and try again")
#             try:
#                 rate.sleep()
#             except rospy.ROSInterruptException:
#                 # This is to avoid error when world is rested, time when backwards.
#                 pass
#         rospy.logdebug("diving_cell_rear Publisher Connected")
#
#         while self.diving_cell_front.get_num_connections() == 0 and not rospy.is_shutdown():
#             rospy.logdebug("No susbribers to diving_cell_front yet so we wait and try again")
#             try:
#                 rate.sleep()
#             except rospy.ROSInterruptException:
#                 # This is to avoid error when world is rested, time when backwards.
#                 pass
#         rospy.logdebug("diving_cell_front Publisher Connected")
#
#
#         rospy.logdebug("All Publishers READY")
#
#
#     # Methods that the TrainingEnvironment will need to define here as virtual
#     # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
#     # TrainingEnvironment.
#     # ----------------------------
#     def _set_init_pose(self):
#         """Sets the Robot in its init pose
#         """
#         raise NotImplementedError()
#
#     def _init_env_variables(self):
#         """Inits variables needed to be initialised each time we reset at the start
#         of an episode.
#         """
#         raise NotImplementedError()
#
#     def _compute_reward(self, observations, done):
#         """Calculates the reward to give based on the observations given.
#         """
#         raise NotImplementedError()
#
#     def _set_action(self, action):
#         """Applies the given action to the simulation.
#         """
#         raise NotImplementedError()
#
#     def _get_obs(self):
#         raise NotImplementedError()
#
#     def _is_done(self, observations):
#         """Checks if episode done based on observations given.
#         """
#         raise NotImplementedError()
#
#     # Methods that the TrainingEnvironment will need.
#     # ----------------------------
#     def set_thruster_speed(self, thruster_rpms, time_sleep):
#         """
#         It will set the speed of each thruster of the deepleng.
#         """
#
#         x_thruster_rpm = FloatStamped()
#         y_thruster_rear_rpm = FloatStamped()
#         y_thruster_front_rpm = FloatStamped()
#         diving_cell_rear_rpm = FloatStamped()
#         diving_cell_front_rpm = FloatStamped()
#
#         x_thruster_rpm.data = thruster_rpms[0]
#         y_thruster_rear_rpm.data = thruster_rpms[1]
#         y_thruster_front_rpm.data = thruster_rpms[2]
#         diving_cell_rear_rpm.data = thruster_rpms[3]
#         diving_cell_front_rpm.data = thruster_rpms[4]
#
#         self.x_thruster.publish(x_thruster_rpm)
#         self.y_thruster_rear.publish(y_thruster_rear_rpm)
#         self.y_thruster_front.publish(y_thruster_front_rpm)
#         self.diving_cell_front .publish(diving_cell_front_rpm)
#         self.diving_cell_rear.publish(diving_cell_rear_rpm)
#
#         self.wait_time_for_execute_movement(time_sleep)
#
#     def wait_time_for_execute_movement(self, time_sleep):
#         """
#         Because this Wamv position is global, we really dont have
#         a way to know if its moving in the direction desired, because it would need
#         to evaluate the diference in position and speed on the local reference.
#         """
#         time.sleep(time_sleep)
#
#     def get_pose(self):
#         return self.auv_pose
#
