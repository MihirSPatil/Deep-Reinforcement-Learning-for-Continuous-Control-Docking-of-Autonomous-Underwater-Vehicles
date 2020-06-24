#!/usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped

class pose_mode:

  def __init__(self):
    self.pose_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)
    self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input',FloatStamped,queue_size=2)
    self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input',FloatStamped,queue_size=2)
    self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input',FloatStamped,queue_size=2)
    self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input',FloatStamped,queue_size=2)
    self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input',FloatStamped,queue_size=2)

  def pose_callback(self, data):
      # names of the models
      # print("models: {}".format(data.name[0]))

      #poses of each of the models
      # print("poses: {}".format(data.pose[0]))

      #angular and linear velocities of the models
      # print("velocities: {}".format(data.twist[0]))

      x_thruster_rpm = FloatStamped()
      y_thruster_rear_rpm = FloatStamped()
      y_thruster_front_rpm = FloatStamped()
      diving_cell_rear_rpm = FloatStamped()
      diving_cell_front_rpm = FloatStamped()

      x_thruster_rpm.data = 10.0
      y_thruster_rear_rpm.data = 0
      y_thruster_front_rpm.data = 0
      diving_cell_rear_rpm.data = 0
      diving_cell_front_rpm.data = 0


      self.x_thruster.publish(x_thruster_rpm)

def main():
    rospy.init_node('pose_mode')
    pm = pose_mode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
