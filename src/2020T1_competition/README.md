# 2020 UBC Parking competition

The repository contains the following ROS packages:

| Folders         | Description      |
|:--------------- |:---------------- |
| enph353_gazebo  | describes simulation world |
| enph353_npcs    | describes and controls pedestrian and Ford truck |
| enph353_utils   | contains competition startup scripts |
| adeept_awr      | describes and controls simulated Adeept AWR robot |
| adeept_awr_ros_driver | controls real world Adeept AWR robot |

## Installation instructions:
** Prerequisites: Ubuntu 18.04 with ROS Melodic installed **

* If you **do not** have a workspace already create one in your home directory.
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

* Clone the repository into a catkin workspace src folder.
```
git clone https://github.com/ENPH353/2020T1_competition.git
```

* Build the packages
```
roscd; cd ..
catkin_make
```

* Source the environment
```
source devel/setup.bash
```

* Start the simulated world
```
cd src/2020T1_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpg
```
The available options are:

| Option | Description      |
|:-------|:---------------- |
| -v     | spawn vehicle    |
| -p     | spawn pedestrian |
| -g     | generate new license plates |

* Start the score tracking app
Open a new tab in the current terminal window by pressing Ctrl+Shift+T 
The new terminal should already be in:
```
~/ros_ws/src/2020T1_competition/enph353/enph353_utils/scripts
```
Launch the score tracking app:
```
./score_tracker.py
```
