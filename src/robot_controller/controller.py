#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from tensorflow import keras

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

# tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

MODEL_PATH = '../drive_data_collect/drive_model'

'''
Reads camera data and robot commands and saves along with linear and angular movement to use as training data.
'''
class RobotController():

    def __init__(self):
        self.bridge = CvBridge()
        self.drive_model = keras.models.load_model(MODEL_PATH)

        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        
    def callback(self, data):
        move = Twist()

        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image = cv2.resize(cv_image, (640, 360))
        # cv2.imshow("image rescaled",cv_image)
        # cv2.waitKey(1)
        cv_image = cv_image/255
        
        
        drive_predict = None
        img_aug = np.expand_dims(cv_image, axis=0)
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            drive_predict = self.drive_model.predict(img_aug)[0]

        # cap linear movement for now
        move.linear.x = min(drive_predict[0], 0.75)
        move.angular.z = drive_predict[1]

        self.cmd_vel_pub.publish(move)
        print("lin:{}, ang:{}".format(move.linear.x, move.angular.z))

        rospy.sleep(0.05)

if __name__ == '__main__':
    rospy.init_node('robot_controller')
    dc = RobotController()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass