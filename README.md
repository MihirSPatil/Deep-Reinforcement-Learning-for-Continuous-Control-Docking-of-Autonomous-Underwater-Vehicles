# Thesis_Code
Code for the thesis

The package deepleng_description causes gazebo 7 to crash often, I don't know the reason why.

Hence the package deepleng_description_simplified was created, which causes less frequent crashes. It containes all the joints, links and actuators in a single base.xacro file.

camera_snippets.xacro file should go into uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins/urdf/
