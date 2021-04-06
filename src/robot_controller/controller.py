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

MODEL_PATH_OUTER = '../drive_data_collect/drive_model'
MODEL_PATH_INNER = '../drive_data_collect/drive_model_inner'

def check_left(img,x1,x2):
    img_diff = np.amax(img[x1:x2,100:600,:], axis=2) - np.amin(img[x1:x2,100:600,:], axis=2)
    img_mean = np.mean(img[x1:x2,100:600,:], axis=2)
    grey_on_line = (img_diff < 10) * (img_mean < 100)
    return np.sum(grey_on_line == 0) < 10

'''
Reads camera data and robot commands and saves along with linear and angular movement to use as training data.
'''
class RobotController():

    def __init__(self):
        self.bridge = CvBridge()
        self.drive_model_outer = keras.models.load_model(MODEL_PATH_OUTER)
        self.drive_model_inner = keras.models.load_model(MODEL_PATH_INNER)

        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

        # TODO: set this variable after finding all outer plates
        self.turn_in = False
        self.inner_loop = False

        # temp
        self.counter = 0
        
    def callback(self, data):
        move = Twist()

        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image_small = cv2.resize(cv_image, (640, 360))
        # cv2.imshow("image rescaled",cv_image)
        # cv2.waitKey(1)

        # check turn-in conditions
        turn_in_available = check_left(cv_image, 448, 452)

        # temp
        if self.counter == 85:
            self.turn_in = True
            print("Turn in armed")

        if self.turn_in and turn_in_available:
            move.linear.x = 0.5
            move.angular.z = 2.0

            self.cmd_vel_pub.publish(move)
            # wait for turn to complete
            print("Turning in...")
            rospy.sleep(1.25)

            # wait a bit to kill momentum
            move.linear.x = 0.25
            move.angular.z = 0
            rospy.sleep(0.1)
            print("Done")

            self.turn_in = False
            self.inner_loop = True
        else:
            # predict using neural net
            cv_image_norm = cv_image_small/255
            
            drive_predict = None
            img_aug = np.expand_dims(cv_image_norm, axis=0)
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                if self.inner_loop:
                    drive_predict = self.drive_model_inner.predict(img_aug)[0]
                else:
                    drive_predict = self.drive_model_outer.predict(img_aug)[0]

            # cap linear movement to guarantee stability
            move.linear.x = min(drive_predict[0], 0.5)
            move.angular.z = drive_predict[1]

            self.cmd_vel_pub.publish(move)
            print("lin:{}, ang:{}".format(move.linear.x, move.angular.z))

        self.counter += 1

if __name__ == '__main__':
    rospy.init_node('robot_controller')
    dc = RobotController()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass