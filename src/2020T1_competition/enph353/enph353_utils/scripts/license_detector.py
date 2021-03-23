#!/usr/bin/env python
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np
import string
import random
from random import randint
import os
import re

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from std_msgs.msg import String

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding

#TODO
#const plate_model = await tf.loadLayersModel('file://path/to/my-model/model.json');

class Gazebo_Lab06_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        #TODO
        LAUNCH_FILE = '/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/launch/license_detector.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
  
        self.plate_pub = rospy.Publisher('/license_plate', std_msgs/String, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.history = []
        for i in range(0,7):
            self.history.append([])

        self.TeamID = 'ISTHIS'
        self.password = 'working'

        self._seed()

        self.bridge = CvBridge()
  

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        plate_detected = False
        processed_image = []
        HOMOGRAPHY_POINTS = 12
        # cv2.imshow("raw", cv_image)
        # cv2.waitKey(3)
        cv_image_copy = cv_image.copy()
        sift = cv2.xfeatures2d.SIFT_create()
        #TODO
        img = cv2.imread("/content/pictures/aplate_blank.png")

        #plt.imshow(img)
        targetheight, targetwidth, _ = img.shape
        kp_image, desc_image = sift.detectAndCompute(img, None)
        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        frame = cv_image_copy

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        if len(good_points) > HOMOGRAPHY_POINTS:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w, _ = img.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            # print(np.int32(dst))
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            # plt.figure(str(i) + "matched")
            # plt.imshow(frame)
            # plt.show()

            # If plate is identified, stretch it and save it.
            rect_pts = np.int32(dst).reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            # the top-left point has the smallest sum whereas the
            # bottom-right has the largest sum
            s = rect_pts.sum(axis=1)
            rect[0] = rect_pts[np.argmin(s)]
            rect[2] = rect_pts[np.argmax(s)]
            # compute the difference between the points -- the top-right
            # will have the minumum difference and the bottom-left will
            # have the maximum difference
            diff = np.diff(rect_pts, axis=1)
            rect[1] = rect_pts[np.argmin(diff)]
            rect[3] = rect_pts[np.argmax(diff)]

            # now that we have our rectangle of points, let's compute
            # the width of our new image
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # ...and now for the height of our new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            # take the maximum of the width and height values to reach
            # our final dimensions
            # construct our destination points which will be used to
            # map the screen to a top-down, "birds eye" view
            dst = np.array([
                [0, 0],
                [targetwidth - 1, 0],
                [targetwidth - 1, targetheight - 1],
                [0, targetheight - 1]], dtype="float32")
            # calculate the perspective transform matrix and warp
            # the perspective to grab the screen
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(frame, M, (targetwidth, targetheight))

            plt.figure("warped")
            plt.imshow(warp)
            plt.show()
            print(warp.shape)
            processed_image = warp
            plate_detected = True
        else:

            framenotmatched = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

            plt.figure("notmatched")
            plt.imshow(framenotmatched)
            plt.show()

        return processed_image, plate_detected

    def detect_image(self, data):
        cropped_img_set = []
        im = data
        w, h = im.size
        # print(h)
        # print(w)
        index = 0
        # split spot number and plate into seperate images
        #PARKING SPOT
        area = (w / 2, 0, w, 2 * h / 3)
        cim_pre = im.crop(area)
        dim = (int(w / 4), int(h / 3))
        cim = cv2.resize(np.asarray(cim_pre), dim)

        cropped_img_set.append(cim)
        #PLATE
        for index in range(0, 4):
            area = (w * index / 4, 2 * h / 3, w * index / 4 + w / 4, h)
            cim = im.crop(area)


            cropped_img_set.append(cim)
            # plt.figure(i)
            # plt.title(imagename)
            # plt.imshow(cim)
            # plt.show()

        # plt.figure(i+4)
        # plt.title(imagename)
        # plt.imshow(cim)
        # plt.show()
        def labelimage(plateID):
            encodingkey = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                           11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                           21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
                           31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}


            label = encodingkey[plateID]
            return label
        predicted = ""

        for i in range(0,len(cropped_img_set)):
            img = cropped_img_set[i]
            img_aug = np.expand_dims(img, axis=0)
            #predicted+= labelimage(np.argmax(plate_model.predict(img_aug)[0]))
            predicted += "1"

        return predicted

    def most_common_plate(self, index):
        return Counter(self.history[index]).most_common(1)

    def step(self):

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        processed_image, plate_detected = self.process_image(data)
        if plate_detected:
            unparsed_string = self.detect_image(processed_image)
            if int(unparsed_string[0]) < 9 & int(unparsed_string[0]) > 0:
                self.history[int(unparsed_string[0]-1)].append(unparsed_string)
                temp_string = most_common_plate(int(unparsed_string[0]))
                self.plate_pub.publish(str(self.TeamID + "," + self.password + ","+temp_string[0]+"," + temp_string[1:] ))
