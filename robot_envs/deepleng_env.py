import numpy as np
import rospy
import time
from openai_ros import robot_gazebo_env
from gazebo_msgs.msg import ModelStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped



class DeeplengEnv(robot_gazebo_env.RobotGazeboEnv):

    def __init__(self):
        """
        Initializes a new DeeplengEnv environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesnt flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /deepleng/deepleng/camera/camera_image: Camera feed from the on-board camera
        * /gazebo/model_states: contains the current pose and velocity of the auv
        
        Actuators Topic List: 
        * /deepleng/thrusters/0/input: publish speed of the back thruster
        * /deepleng/thrusters/1/input: publish speed of the rear y-thruster
        * /deepleng/thrusters/2/input: publish speed of the front y-thruster
        * /deepleng/thrusters/3/input: publish speed of the front diving cell
        * /deepleng/thrusters/4/input: publish speed of the rear diving cell

        
        Args:
        """
        rospy.logdebug("Start DeeplengEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(DeeplengEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="WORLD")



        rospy.logdebug("DeeplengEnv unpause1...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        
        self._check_all_systems_ready()


        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._pose_callback)

        self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input', FloatStamped, queue_size=1)
        self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input', FloatStamped, queue_size=1)
        self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input', FloatStamped, queue_size=1)
        self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input', FloatStamped, queue_size=1)
        self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input', FloatStamped, queue_size=1)

        # self.publishers_array = []
        # self._cmd_drive_pub = rospy.Publisher('/cmd_drive', UsvDrive, queue_size=1)
        
        # self.publishers_array.append(self.x_thruster)
        # self.publishers_array.append(self.y_thruster_rear)
        # self.publishers_array.append(self.y_thruster_front)
        # self.publishers_array.append(self.diving_cell_front)
        # self.publishers_array.append(self.diving_cell_rear)

        # self._check_all_publishers_ready()

        self._check_pub_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished DeeplengEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("DeeplengEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END DeeplengEnv _check_all_systems_ready...")
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_pose_ready()
        rospy.logdebug("ALL SENSORS READY")

        
    def _check_pose_ready(self):
        self.auv_pose = None
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        while self.auv_pose is None and not rospy.is_shutdown():
            try:
                self.auv_pose = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=1.0)
                rospy.logdebug("Current /gazebo/model_states READY=>")

            except:
                rospy.logerr("Current /gazebo/model_states not ready yet, retrying for getting auv pose")
        return self.auv_pose
        
        
    
    def _pose_callback(self, data):
        self.auv_pose = data.pose[-1]
        # names of the models
        # print("models: {}".format(data.name[0]))

        # poses of each of the models
        # print("poses: {}".format(data.pose[0]))
    
    # def _check_all_publishers_ready(self):
    #     """
    #     Checks that all the publishers are working
    #     :return:
    #     """
    #     rospy.logdebug("START ALL SENSORS READY")
    #     for publisher_object in self.publishers_array:
    #         self._check_pub_connection(publisher_object)
    #     rospy.logdebug("ALL SENSORS READY")

    def _check_pub_connection(self):

        rate = rospy.Rate(10)  # 10hz
        while self.x_thruster.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to x_thruster yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("x_thruster Publisher Connected")

        while self.y_thruster_rear.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to y_thruster_rear yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("y_thruster_rear Publisher Connected")

        while self.y_thruster_front.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to y_thruster_front yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("y_thruster_front Publisher Connected")

        while self.diving_cell_rear.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to diving_cell_rear yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("diving_cell_rear Publisher Connected")

        while self.diving_cell_front.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to diving_cell_front yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("diving_cell_front Publisher Connected")


        rospy.logdebug("All Publishers READY")
        
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
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
        self.diving_cell_front .publish(diving_cell_front_rpm)
        self.diving_cell_rear.publish(diving_cell_rear_rpm)

        self.wait_time_for_execute_movement(time_sleep)
    
    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the diference in position and speed on the local reference.
        """
        time.sleep(time_sleep)
    
    def get_pose(self):
        return self.auv_pose

