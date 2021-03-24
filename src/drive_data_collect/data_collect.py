#!/usr/bin/env python
#import roslib
#roslib.load_manifest('enph353')

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

IMAGE_DIR_PATH = '~/Code/ML/src/drive_data_collect/image_data/'

'''
Reads camera data and robot commands and saves along with linear and angular movement to use as training data.
'''
class DataCollector():

    def __init__(self):
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.counter = 0

        self.bridge = CvBridge()
        
    def callback(self, data):
        cmd_vel = rospy.wait_for_message('/R1/cmd_vel', Twist, timeout=0.5)

        vel = cmd_vel.linear.x
        ang = cmd_vel.angular.z
        
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

        out = cv2.imwrite('/home/quinn/Code/ML/src/drive_data_collect/image_data/driving_{}_v{}_a{}.png'.format(self.counter, np.round(vel, 4), np.round(ang, 4)), cv_image)
        print('status: {}  saving img {} {} {}'.format(out, self.counter, np.round(vel, 4), np.round(ang, 4)))
        self.counter += 1


if __name__ == '__main__':
    rospy.init_node('data_collection')
    dc = DataCollector()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
