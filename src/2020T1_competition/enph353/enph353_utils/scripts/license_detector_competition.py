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
import sys
import re
import tensorflow
import keyboard

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from std_msgs.msg import String

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding
sess1 = tensorflow.Session()
graph1 = tensorflow.get_default_graph()
set_session(sess1)


class License_Detector():

    def __init__(self, arg):
        # Launch the simulation with the given launchfile name
        #TODO
        # LAUNCH_FILE = '/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/launch/license_detector_competition.launch'
        # gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.TeamID = 'ISTHIS'
        self.password = 'working'
        self.bridge = CvBridge()

        self.history = {'1' : [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8' : []}
        # initialize game
        rospy.init_node('license_publisher')
        self.plate_model = tensorflow.keras.models.load_model("/home/fizzer/content_353/detection_CNN.json")




        self.plate_pub = rospy.Publisher('/license_plate', String, queue_size=10)

        #start step loop
        sleep(1)
        self.image_feed = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.step, queue_size=3)

        if arg == "-stop":
            self.plate_pub.publish(str(self.TeamID + "," + self.password + ",-1,0000"))
            print("------------Stopped run------------")
        else:
            self.plate_pub.publish(str(self.TeamID + "," + self.password + ",0,0000"))
            print("------------Started run------------")


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
        HOMOGRAPHY_POINTS = 15
        # cv2.imshow("raw", cv_image)
        # cv2.waitKey(3)
        cv_image_copy = cv_image.copy()
        sift = cv2.xfeatures2d.SIFT_create()

        img = cv2.imread("/home/fizzer/content_353/aplate_blank_flat.png")

        # #plt.imshow(img)
        targetheight = img.shape[0]
        targetwidth = img.shape[1]

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
            if m.distance < 0.68 * n.distance:
                good_points.append(m)

        if len(good_points) > HOMOGRAPHY_POINTS:
            query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
            train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            # Perspective transform
            h, w, _ = img.shape
            print("SIFT IMAGE: " + str(img.shape))
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
            cv2.imshow("detected", warp)
            cv2.waitKey(3)
            # plt.figure("warped")
            # plt.imshow(warp)
            # plt.show()
            #print(warp.shape)
            processed_image = warp
            plate_detected = True
        else:

            framenotmatched = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
            cv2.imshow("NOT!!! detected", framenotmatched)
            cv2.waitKey(3)
            # plt.figure("notmatched")
            # plt.imshow(framenotmatched)
            # plt.show()

        return processed_image, plate_detected

    def detect_image(self, data):
        cropped_img_set = []
        im = data
        w = im.shape[0]
        h = im.shape[1]
        # print(h)
        # print(w)
        index = 0
        # split spot number and plate into seperate images
        #PARKING SPOT
        #area = (w / 2, 0, w, 2.0 * h / 3)

        cim_pre = im[int(w/2):w,0:int(2.0*h/3),:]
        #dim = (int(w / 4), int(h / 3))
        dim = (int(150), int(299))
        cim = cv2.resize(np.asarray(cim_pre), dim)

        cropped_img_set.append(cim)
        #PLATE
        for index in range(0, 4):
            #area = (w * index / 4, 2.0 * h / 3, w * index / 4 + w / 4, h)
            cim = im[int(w*index/4):int(w*index/4+w/4),int(2.0*h/3):h,:]
            cim = cv2.resize(np.asarray(cim), dim)


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
            img = np.asarray(cropped_img_set[i])
            img_aug = np.expand_dims(img, axis=0)
            cv2.imshow("abouttabedetected", img)
            cv2.waitKey(3)

            with graph1.as_default():
                set_session(sess1)
                predicted_label = self.plate_model.predict(img_aug)[0]
            print(predicted_label)
            predicted += labelimage(np.argmax(predicted_label))


        return predicted

    def most_common_plate(self, number):
        return Counter(self.history[number]).most_common(1)[0]



    def step(self, data):
        processed_image, plate_detected = self.process_image(data)
        if plate_detected:
            unparsed_string = self.detect_image(processed_image)
            print(unparsed_string)
            print(unparsed_string[0])


            try:
                self.history[unparsed_string[0]].append(unparsed_string)
                temp_string = self.most_common_plate(unparsed_string[0])[0]
                print(temp_string)
                self.plate_pub.publish(str(self.TeamID + "," + self.password + ","+temp_string[0]+"," + temp_string[1:]))
                print("published: "+ str(self.TeamID + "," + self.password + "," + temp_string[0] + "," + temp_string[1:]))
                #self.plate_pub.publish(str(self.TeamID + "," + self.password + ",-1,0000"))
            except KeyError:
                pass


if __name__ == '__main__':
    #rospy.init_node('license_detector_competition')
    if len(sys.argv) > 1:
        ld = License_Detector(sys.argv[1])
    else:
        ld = License_Detector("start")
    try:
        rospy.spin()
        try:
            pass
        except KeyboardInterrupt:
            ld.plate_pub.publish(str(ld.TeamID + "," + ld.password + ",-1,0000"))
            exit()

    except rospy.ROSInterruptException:
        pass
