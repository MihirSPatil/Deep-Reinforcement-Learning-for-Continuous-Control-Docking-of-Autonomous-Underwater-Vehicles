#!/usr/bin/env python

import rospy
import cv2 as cv
import numpy as np
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from cv_bridge import CvBridge, CvBridgeError
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped

class camera_mode:

  def __init__(self):
    self.bridge = CvBridge()
    self.camera_sub = rospy.Subscriber('/deepleng/deepleng/camera/camera_image', Image, self.camera_callback)
    self.x_thruster = rospy.Publisher('/deepleng/thrusters/0/input',FloatStamped,queue_size=2)
    self.y_thruster_rear = rospy.Publisher('/deepleng/thrusters/1/input',FloatStamped,queue_size=2)
    self.y_thruster_front = rospy.Publisher('/deepleng/thrusters/2/input',FloatStamped,queue_size=2)
    self.diving_cell_front = rospy.Publisher('/deepleng/thrusters/3/input',FloatStamped,queue_size=2)
    self.diving_cell_rear = rospy.Publisher('/deepleng/thrusters/4/input',FloatStamped,queue_size=2)

  def camera_callback(self, data):
      try:
          frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
      except CvBridgeError as e:
          print(e)

# comment the following lines to prevent a pop-up window of the camera feed
      cv.imshow("Image window", frame)
      self.keystroke = cv.waitKey(1)
      if 32 <= self.keystroke and self.keystroke < 128:
          cc = chr(self.keystroke).lower()
          if cc == 'q':
              rospy.signal_shutdown("User hit q key to quit.")

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
    rospy.init_node('camera_mode')
    cm = camera_mode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
#!/usr/bin/env python
