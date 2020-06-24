#!/usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.msg import ModelStates

class pose_reader:

  def __init__(self):
    self.pose_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)

  def pose_callback(self, data):
      # names of the models
      # print("models: {}".format(data.name[0]))

      #poses of each of the models
      # print("poses: {}".format(data.pose[0]))

      #angular and linear velocities of the models
      # print("velocities: {}".format(data.twist[0]))

def main():
  pr = pose_reader()
  rospy.init_node('pose_reader')
  try:
    rospy.spin()
  except rospy.ROSInterruptException:
    print("Shutting down")


if __name__ == '__main__':
    main()
