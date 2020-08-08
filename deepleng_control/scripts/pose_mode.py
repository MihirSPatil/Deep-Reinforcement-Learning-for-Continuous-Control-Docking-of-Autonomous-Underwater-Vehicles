#!/usr/bin/env python

import message_filters
import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped

class pose_mode:

  def __init__(self):
    self.thrust_sub1 = message_filters.Subscriber("/deepleng/thrusters/0/thrust",FloatStamped)
    self.thrust_sub2 = message_filters.Subscriber("/deepleng/thrusters/1/thrust",FloatStamped)
    self.thrust_sub3 = message_filters.Subscriber("/deepleng/thrusters/2/thrust",FloatStamped)
    self.thrust_sub4 = message_filters.Subscriber("/deepleng/thrusters/3/thrust",FloatStamped)
    self.thrust_sub5 = message_filters.Subscriber("/deepleng/thrusters/4/thrust",FloatStamped)
    self.ts = message_filters.ApproximateTimeSynchronizer([self.thrust_sub1, self.thrust_sub2,self.thrust_sub3, self.thrust_sub4, self.thrust_sub5], 1, 5)
    self.ts.registerCallback(self.thrust_callback)

    self.pose_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)
    self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input',FloatStamped,queue_size=2)
    self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input',FloatStamped,queue_size=2)
    self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input',FloatStamped,queue_size=2)
    self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input',FloatStamped,queue_size=2)
    self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input',FloatStamped,queue_size=2)


  def thrust_callback(self, thruster0, thruster1, thruster2, thruster3, thruster4):
      thrust_vector = np.array([thruster0.data, thruster1.data, thruster2.data, thruster3.data, thruster4.data])
      print("thrust vector: ", thrust_vector)

  def pose_callback(self, data):
      # names of the models
      # print("models: {}".format(data.name[0]))

      #poses of each of the models
      #print("poses: {}".format(data.pose[0].orientation))

      #angular and linear velocities of the models
      # print("velocities: {}".format(data.twist[0]))

      x_thruster_rpm = FloatStamped()
      y_thruster_rear_rpm = FloatStamped()
      y_thruster_front_rpm = FloatStamped()
      diving_cell_rear_rpm = FloatStamped()
      diving_cell_front_rpm = FloatStamped()

      x_thruster_rpm.data = 1.0
      y_thruster_rear_rpm.data = 0
      y_thruster_front_rpm.data = 0
      diving_cell_rear_rpm.data = 0
      diving_cell_front_rpm.data = 0


      #self.x_thruster.publish(x_thruster_rpm)

def main():
    rospy.init_node('pose_mode')
    pm = pose_mode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
