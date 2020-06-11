# Thesis_Code

#### The package deepleng_description causes gazebo 7 to crash often, I don't know the reason why?


#### Hence the package deepleng_description_simplified was created, which causes less frequent crashes. It containes all the joints, links and actuators in a single base.xacro file.


#### camera_snippets.xacro file should go into uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins/urdf/ if being used independently of the uuv_simulator package provided here.


#### Added only the modified uuv_simulator packages to spawn the docking station in a custom world and the package for modified camera parameters of the deepleng auv.


#### Launch the world with roslaunch uuv_gazebo_worlds empty_underwater_world.launch


#### To launch other worlds use either the respective launch file or change the world name in the empty_underwater_world.launch


#### Launch the deepleng auv with roslaunch deepleng_description upload.launch

#### The meshes for docking station are present in uuv_gazebo_worlds/models/deepleng_docking_station.


#### The world designed for use of the docking station is called empty_underwater_docking.


#### Use the rostopic rostopic pub -r 20 /deepleng/thrusters/0/input uuv_gazebo_ros_plugins_msgs/FloatStamped '{header: auto, data: 40.0}' to publish rpm commands to the auv. To publish to different thrusters only the thruster_id(0,1,2) needs to be changed.
