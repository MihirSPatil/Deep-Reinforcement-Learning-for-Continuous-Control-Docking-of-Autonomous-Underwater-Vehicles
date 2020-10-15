 Thesis_Code
 ======

**_This guide assumes that the following packages have been cloned and/or installed by the user_**:
1. _ROS Melodic_
1. _uuv_simulator_
2. _openai_gym_
3. _stable_baselines_
4. _tensorflow 1.14_

### uuv_simulator
***
> Package to simulate the underwater environment

* Added only the modified uuv_simulator packages to spawn the docking station in a custom world and the package for modified camera parameters of the deepleng auv.

* Replace the folders **uuv_gazebo_worlds** and **uuv_sensor_plugins** in the uuv_simulator package with the ones provided here.

* The world designed for use of the docking station is called empty_underwater_docking.


* Launch the world with `roslaunch uuv_gazebo_worlds empty_underwater_world.launch`


* To launch other worlds use either the respective launch file or change the world name in the empty_underwater_world.launch


* Launch the deepleng auv with `roslaunch deepleng_description upload.launch`

### deepleng_description
***
> Package to spawn the deepleng AUV in the simulator

* The meshes for docking station are present in uuv_gazebo_worlds/models/deepleng_docking_station.


* Publish rpm values to the AUV thruster's using:
`rostopic pub -r 20 /deepleng/thrusters/0/input uuv_gazebo_ros_plugins_msgs/FloatStamped '{header: auto, data: 40.0}'` 


 *To publish to different thrusters only the thruster_id(0,1,2) needs to be changed.*


### geometry2
***
> Submodule to aid in ros transforms when running **ros melodic** with **python3**
* This helps overcome the following errors if they occur:
  * [ImportError: dynamic module does not define module export function (PyInit__tf2)](https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/)
  * [https://answers.ros.org/question/340862/importerror-dynamic-module-does-not-define-init-function-init_tf2/](https://answers.ros.org/question/340862/importerror-dynamic-module-does-not-define-init-function-init_tf2/)
  
  
* To build a ros workspace with **catkin_make** and **python3**, please follow the steps in the link below:
  *  * [https://github.com/ros/geometry2/issues/259#issuecomment-353268956](https://github.com/ros/geometry2/issues/259#issuecomment-353268956)
  
  
* Other common errors while building the *ros workspace with catkin_make and python3* are answered in the following links:
  * [https://answers.ros.org/question/260377/no-module-named-defusedxmlxmlrpc/](https://answers.ros.org/question/260377/no-module-named-defusedxmlxmlrpc/)


