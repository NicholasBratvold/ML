#!/bin/bash
rosrun teleop_twist_joy teleop_node _scale_angular:=2.0 /cmd_vel:=/R1/cmd_vel &
rosrun joy joy_node &
