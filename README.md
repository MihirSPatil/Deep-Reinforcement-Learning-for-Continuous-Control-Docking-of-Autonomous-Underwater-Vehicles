 Thesis_Code
 ======

**_This guide assumes that the following packages have been cloned and/or installed by the user_**:
1. _ROS Melodic_
1. _uuv_simulator_
2. _openai_gym 0.17.2_
2. _python 3.6.9_
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


### deepleng_description
***
> Package to spawn the deepleng AUV in the simulator

* The meshes for docking station are present in uuv_gazebo_worlds/models/deepleng_docking_station.


* Publish rpm values to the AUV thruster's using:
`rostopic pub -r 20 /deepleng/thrusters/0/input uuv_gazebo_ros_plugins_msgs/FloatStamped '{header: auto, data: 40.0}'` 
  * >To publish to different thrusters only the thruster_id(0,1,2) needs to be changed.
 
 
 * Launch the deepleng auv with `roslaunch deepleng_description upload.launch`

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


### deepleng_gym
***
> Package containing the openai-gym environment to train and test DRL agents
* The environment can be imported using: `from deepleng_gym.task_envs.deepleng import deepleng_docking`


* Instansiate the environment with: `gym.make('DeeplengDocking-v2')`


* **robot_envs**: contains all the ros and gazebo specific code such as the publishers and subscribers to various rostopics.


* **task_envs**: contains the gym.Env wrapped code that defines the deepleng docking environment.


### deepleng_control
***
> Package containing the python scripts and ros nodes to launch DRL agents from the stable_baselines library
* **scripts**: contains the python scripts where the agent and the environment are defined


* **launch**: contains the launch files used to run the training and evaluation of the DRL agents

* Training can be launched with: `roslaunch deepleng_control start_training.launch`
  * > This will start the simulation, spawn the robot as well as start the training of the DRL agent.
  * Different agents can be trained by changing the node type: `<node pkg="deepleng_control" name="rl_controller" type="stable_baselines_ddpg.py" output="screen"/>` in the launch file.
  
* Inference can be launched with: `roslaunch deepleng_control start_inference.launch`
  * > This will only launch the inference, start the simulation and spawn the AUVseperately using the commands descirbed in the prior sections before starting the inference.
  
  
### [openai_ros](http://wiki.ros.org/openai_ros)
***
> Package that acts as the interface between the gazebo simulator and the openai-gym environment


>[source](http://wiki.ros.org/openai_ros)


How-to:
------

1. Create a python3 virtual_env containing atleast:
  * _openai_gym 0.17.2_
  * _stable_baselines_
  * _tensorflow 1.14_
  * _numpy_
  * _scipy_
  * _defusedxml_
  
2. Create a **ROS melodic workspace** using the folders present here _(modify the uuv_simulator package as described earlier)_ that is built with **catkin build** and **python2**.


3.  Create a seperate **ROS melodic workspace** using the folders present here _(modify the uuv_simulator package as described earlier)_ and build it with **catkin_make** and **python3**


4. Source the virtual_env before sourcing the ROS workspaces in the following order:
  1. Source ROS workspace built with **python2** and **catkin build**
  2. Source ROS workspace built with **python3** and **catkin_make**
  
5. Set _experiment_name_ in the **set_env_variables.sh** script and source it.


6. Launch training or inference using the commands described in the prior sections.



> To-do: add folder_structure later
