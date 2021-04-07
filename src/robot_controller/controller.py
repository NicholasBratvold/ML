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

from enum import Enum

# tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

MODEL_PATH_OUTER = '../drive_data_collect/drive_model'
MODEL_PATH_INNER = '../drive_data_collect/drive_model_inner'

# constants
CROSSWALK_ROW = 550
CROSSWALK_MIN_COL = 100
CROSSWALK_MAX_COL = 1100

PED_MIN_ROW = 300
PED_MAX_ROW = 500
PED_MIN_COL = 370
PED_MAX_COL = 800

CAR_MIN_ROW = 300
CAR_MAX_ROW = 600
CAR_MIN_COL = 300
CAR_MAX_COL = 900

CAR_THRESHOLD = 1000
CAR_TURN_IN_THRESHOLD = 500

'''
Current state of the robot.
'''
class State(Enum):
    outer_loop = 0
    stop_crosswalk = 1
    turn_intersection_ready = 2
    turn_intersection = 3
    stop_intersection = 4
    turn_inner = 5
    inner_loop = 6
    stop_inner = 7

def check_grey(img, x1, y1, x2, y2):
    img_diff = np.amax(img[x1:x2,y1:y2,:], axis=2) - np.amin(img[x1:x2,y1:y2,:], axis=2)
    img_mean = np.mean(img[x1:x2,y1:y2,:], axis=2)
    grey_on_line = (img_diff < 10) * (img_mean < 100)
    return np.sum(grey_on_line == 0) < 50

def check_crosswalk(img):
    return np.sum(img[CROSSWALK_ROW:,CROSSWALK_MIN_COL:CROSSWALK_MAX_COL,0] - np.mean(img[CROSSWALK_ROW:,CROSSWALK_MIN_COL:CROSSWALK_MAX_COL,:], axis=2) > 150) > 10000

def check_pedestrian(img):
    rel_blue = img[PED_MIN_ROW:PED_MAX_ROW,PED_MIN_COL:PED_MAX_COL,2] - np.mean(img[ PED_MIN_ROW:PED_MAX_ROW, PED_MIN_COL:PED_MAX_COL], axis=2)
    return np.sum((rel_blue > 5) & (rel_blue < 25)) > 200

def check_car(img, x1, y1, x2, y2, threshold=CAR_THRESHOLD):
    img = img[x1:x2, y1:y2]
    img_diff = np.amax(img, axis=2) - np.amin(img, axis=2)
    img_mean = np.mean(img, axis=2)

    dark = (img_diff == 0) * (img_mean < 30)
    return np.sum(dark) > threshold

'''
Reads camera data and robot commands and saves along with linear and angular movement to use as training data.
'''
class RobotController():

    def __init__(self):
        self.bridge = CvBridge()
        self.drive_model_outer = keras.models.load_model(MODEL_PATH_OUTER)
        self.drive_model_inner = keras.models.load_model(MODEL_PATH_INNER)

        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.view_pub = rospy.Publisher('/robot_view', Image, queue_size=1)

        # this time gets used by states whenever they need to wait for some time
        self.last_time = rospy.get_time()
        self.last_crosswalk = rospy.get_time()

        self.state = State.outer_loop

        rospy.sleep(1)
        self._pub_move(0.4, 1.5)
        rospy.sleep(1)
        
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.callback)

        
    def callback(self, data):
        cur_time = rospy.get_time()

        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image_small = cv2.resize(cv_image, (640, 360))

        # image published to robot_view
        out_image = cv_image

        # main state loop
        if self.state == State.outer_loop:
            # check for crosswalk
            cv2.rectangle(out_image, (CROSSWALK_MIN_COL, 720), (CROSSWALK_MAX_COL, CROSSWALK_ROW), (0, 0, 255), 3)
            if check_crosswalk(cv_image) and cur_time - self.last_crosswalk > 4:
                print('Stopping for the nice man.')
                self.state = State.stop_crosswalk
                self._pub_stop()
                # wait a lil
                rospy.sleep(0.25)
            else:
                # TODO: license plate reading

                # TODO: check if outer license plates have been read
                # temp
                if cur_time - self.last_time >= 30:
                    self.state = State.turn_intersection_ready
                    print('Turn in ready.')

                self._pub_drive_prediction(cv_image_small, is_inner=False)

        elif self.state == State.stop_crosswalk:
            cv2.rectangle(out_image, (PED_MIN_COL, PED_MIN_ROW), (PED_MAX_COL, PED_MAX_ROW), (255, 255, 0), 3)
            if check_pedestrian(cv_image):
                pass
            else:
                print('Continuing...')
                self.state = State.outer_loop
                self._pub_drive_prediction(cv_image_small, is_inner=False)
                self.last_crosswalk = rospy.get_time()

        elif self.state == State.turn_intersection_ready:
            # check turn-in conditions
            turn_in_available = check_grey(cv_image, 445, 200, 455, 700) and check_grey(cv_image, 400, 695, 450, 705)
            cv2.line(out_image, (100, 450), (700, 450), (255,0, 0), 10)
            cv2.line(out_image, (700, 450), (700, 400), (255,0, 0), 10)

            if turn_in_available:
                self.state = State.turn_intersection

                print('Turning in...')
                self._pub_move(0.5, 2.0)

                self.last_time = rospy.get_time()
            else:
                self._pub_drive_prediction(cv_image_small, is_inner=False)

        elif self.state == State.turn_intersection:
            # TODO: possibly change to using camera to decide when to stop
            if cur_time - self.last_time >= 0.85:
                self._pub_stop()

                self.state = State.stop_intersection
                rospy.sleep(0.5)
                self.last_time = rospy.get_time()
            else:
                pass

        elif self.state == State.stop_intersection:
            cv2.rectangle(out_image, (0, CAR_MIN_ROW), (CAR_MAX_COL, CAR_MAX_ROW), (255, 0, 255), 3)
            if check_car(cv_image, CAR_MIN_ROW, 0, CAR_MAX_ROW, CAR_MAX_COL, CAR_TURN_IN_THRESHOLD) == False:
                self._pub_move(0.35, 2)
                self.state = State.turn_inner
                self.last_time = rospy.get_time()
            else:
                pass

        elif self.state == State.turn_inner:
            if cur_time - self.last_time >= 0.75:
                self.state = State.inner_loop
                self._pub_drive_prediction(cv_image_small, is_inner=True)
                print("Done")
            else:
                pass

        elif self.state == State.inner_loop:
            # TODO: stop timer once all plates are found

            cv2.rectangle(out_image, (CAR_MIN_COL, CAR_MIN_ROW), (CAR_MAX_COL, CAR_MAX_ROW), (255, 0, 255), 3)
            if check_car(cv_image, CAR_MIN_ROW, CAR_MIN_COL, CAR_MAX_ROW, CAR_MAX_COL):
                self._pub_stop()
                self.state = State.stop_inner
                print('Stopping for mean car.')
                # zzz while car moves away
                rospy.sleep(2)
            else:
                self._pub_drive_prediction(cv_image_small, is_inner=True)

        elif self.state == State.stop_inner:
            cv2.rectangle(out_image, (CAR_MIN_COL, CAR_MIN_ROW), (CAR_MAX_COL, CAR_MAX_ROW), (255, 0, 255), 3)
            if check_car(cv_image, CAR_MIN_ROW, CAR_MIN_COL, CAR_MAX_ROW, CAR_MAX_COL):
                pass
            else:
                self._pub_drive_prediction(cv_image_small, is_inner=True)
                self.state = State.inner_loop
                print('Continuing...')

        out_image = self.bridge.cv2_to_imgmsg(out_image, encoding="rgb8")

        self.view_pub.publish(out_image)
    
    ''' Get prediction from neural net and publish a movement for it. '''
    def _pub_drive_prediction(self, cv_image_small, is_inner):
            cv_image_norm = cv_image_small/255
            
            drive_predict = None
            img_aug = np.expand_dims(cv_image_norm, axis=0)
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                if is_inner:
                    drive_predict = self.drive_model_inner.predict(img_aug)[0]
                else:
                    drive_predict = self.drive_model_outer.predict(img_aug)[0]

            self._pub_move(min(drive_predict[0], 0.4), drive_predict[1])
    
    ''' Publish a move command with specified linear and angular velocity. '''
    def _pub_move(self, lin, ang):
            move = Twist()
            move.linear.x = lin
            move.angular.z = ang

            self.cmd_vel_pub.publish(move)
            print("lin:{}, ang:{}".format(move.linear.x, move.angular.z))

    ''' Stop robot, slowing a bit to account for momentum. '''
    def _pub_stop(self):
        self._pub_move(0.35, 0)
        rospy.sleep(0.1)
        self._pub_move(0.25, 0)
        rospy.sleep(0.1)
        self._pub_move(0.15, 0)
        rospy.sleep(0.1)
        self._pub_move(0.05, 0.0)
        rospy.sleep(0.1)
        self._pub_move(0.0, 0.0)


if __name__ == '__main__':
    rospy.init_node('robot_controller')
    dc = RobotController()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass