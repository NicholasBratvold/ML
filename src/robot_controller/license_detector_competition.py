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
"""
Change self.path variable
There must be a letter_detection_CNN and a number_detection_CNN in path directory.

This script currently starts the timer but needs to be removed.
"""
class License_Detector():

    def __init__(self, arg):
        # Launch the simulation with the given launchfile name
        #TODO
        # LAUNCH_FILE = '/home/fizzer/ros_ws/src/2020T1_competition/enph353/enph353_utils/launch/license_detector_competition.launch'
        # gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.TeamID = 'ISTHIS'
        self.password = 'working'
        self.bridge = CvBridge()
        self.path_dir =  os.path.dirname(os.path.realpath(__file__)) + "/"

        #this path should be "~/ros_ws/src/licensegeneration" or similiar.
        self.path = "../../../../license_generation/"
        self.history = {'1' : [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8' : []}
        # initialize game
        rospy.init_node('license_publisher')
        self.plate_model_numbers = tensorflow.keras.models.load_model(self.path + "number_detection_CNN")
        self.plate_model_letters = tensorflow.keras.models.load_model(self.path + "letter_detection_CNN")
        self.process_counter = 0
        #Set this to false to remove visuals
        self.visuals = False


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

        detected = False
        dst = []
        img_orig = cv_image

        if self.visuals:
            cv2.imshow("raw", cv_image)
            cv2.waitKey(3)


        row, col, plane = img_orig.shape

        # crop image
        img = img_orig[int(row * 2 / 5):int(row * 7 / 8), :, :].copy()
        img_crop = img.copy()

        # Threshold in hsv
        # THESE ARE REALLY GOOD DONT CHANGE
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        lower_spot_grey = np.array([0, 0, 91])
        upper_spot_grey = np.array([30, 30, 210])

        lower_plate_grey = np.array([99, 1, 80])
        upper_plate_grey = np.array([120, 30, 180])

        # preparing the mask to overlay
        mask_plate = cv2.inRange(hsv, lower_plate_grey, upper_plate_grey)
        mask_spot = cv2.inRange(hsv, lower_spot_grey, upper_spot_grey)

        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all unwanted regions
        plate = cv2.bitwise_and(img_crop, img_crop, mask=mask_plate)
        spot = cv2.bitwise_and(img_crop, img_crop, mask=mask_spot)

        # add plate and spot together
        result = plate + spot
        if self.visuals:
            cv2.imshow("hsv altercation", result)
            cv2.waitKey(3)

        # mask and threshold
        result_grey = result[:, :, 2]
        thresh = 30
        ret, result_binary = cv2.threshold(result_grey, thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        result_binary = cv2.morphologyEx(result_binary, cv2.MORPH_CLOSE, kernel)
        result_binary = cv2.morphologyEx(result_binary, cv2.MORPH_OPEN, kernel)


        # edge detext
        edged = cv2.Canny(result_binary, 0, 200)

        if self.visuals:
            cv2.imshow("first mask", edged)
            cv2.waitKey(3)

        # compute contours
        _, contours, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        if len(contours) > 0:
            contours_poly = [None] * len(contours)

            contours_poly[0] = cv2.approxPolyDP(contours[0], 20, True)


            # make sure its square-ish and nothing else!
            if len(contours_poly[0]) == 4:

                rectPoints = contours_poly[0]

                rect_pts = np.int32(rectPoints).reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                # the top-left point has the smallest sum whereas the
                # bottom-right has the largest sum
                s = rect_pts.sum(axis=1)
                rect[0] = rect_pts[np.argmin(s)]
                rect[2] = rect_pts[np.argmax(s)]
                tx, ty = rect[0]
                bx, by = rect[2]

                # make sure quality of image is decent
                minheight = 100
                minwidth = 60
                if (bx - tx) * (by - ty) > minwidth*minheight:
                    # compute the difference between the points -- the top-right
                    # will have the minumum difference and the bottom-left will
                    # have the maximum difference
                    diff = np.diff(rect_pts, axis=1)
                    rect[1] = rect_pts[np.argmin(diff)]
                    rect[3] = rect_pts[np.argmax(diff)]
                    targetwidth = 300
                    targetheight = 900
                    dst = np.array([
                        [0, 0],
                        [targetwidth - 1, 0],
                        [targetwidth - 1, targetheight - 1],
                        [0, targetheight - 1]], dtype="float32")

                    # calculate the perspective transform matrix and warp
                    # the perspective to grab the plate
                    M = cv2.getPerspectiveTransform(rect, dst)
                    dst = cv2.warpPerspective(img_crop, M, (targetwidth, targetheight))

                    # detected is true when a
                    detected = True

                    #standard deviation of image incase garbage is picked up
                    dst_mean = np.mean(dst)
                    std_dst = np.std(dst - dst_mean)
                    #print(std_dst)

                    #Actual plates standard deviation range from 10 to 37
                    if std_dst < 8 or std_dst > 40:
                        detected = False



        processed_image = dst
        plate_detected = detected
        return processed_image, plate_detected


    def cropper(self, img):
        h,w,_ = img.shape
        # print(w)
        # print(h)
        dim = (50, 100)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.visuals:
            cv2.imshow("bw", img)
            cv2.waitKey(3)
        thresh = 64



        # cv2.imwrite(os.path.join(self.path + "warpedpictures/",
        #                          "plate{}.png".format(self.process_counter)), img)
        s = img[int(h / 2.6):int( h / 1.6), int(w / 2):w]
        a0 = img[int(h / 1.44):int(h / 1.16), 0:int(w / 4.0)]
        a1 = img[int(h / 1.44):int(h / 1.16), int(w / 4.0):int(w / 2.0)]
        n0 = img[int(h / 1.44):int(h / 1.16), int(w / 2.0):int(3 * w / 4.0)]
        n1 = img[int(h / 1.44):int(h / 1.16), int(3 * w / 4.0):w]
        s = cv2.resize(s, dim)
        a0 = cv2.resize(a0, dim)
        a1 = cv2.resize(a1, dim)
        n0 = cv2.resize(n0, dim)
        n1 = cv2.resize(n1, dim)
        kernel = np.ones((5, 5), np.uint8)

        # cv2.imshow("b", s)
        # cv2.waitKey(3)
        # cv2.imshow("w", a1)
        # cv2.waitKey(3)

        s = cv2.adaptiveThreshold(s, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        a0 = cv2.adaptiveThreshold(a0, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        a0 = cv2.morphologyEx(a0, cv2.MORPH_OPEN, kernel)
        a1 = cv2.adaptiveThreshold(a1, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        a1 = cv2.morphologyEx(a1, cv2.MORPH_OPEN, kernel)
        n0 = cv2.adaptiveThreshold(n0, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        n0 = cv2.morphologyEx(n0, cv2.MORPH_OPEN, kernel)
        n1 = cv2.adaptiveThreshold(n1, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        n1 = cv2.morphologyEx(n1, cv2.MORPH_OPEN, kernel)
        # s = cv2.Canny(s, 0, 255)
        # a0 = cv2.Canny(a0, 0, 255)
        # a1 = cv2.Canny(a1, 0, 255)
        # n0 = cv2.Canny(n0, 0, 255)
        # # n1 = cv2.Canny(n1, 0, 255)

        if self.visuals:
            s_copy = cv2.resize(s, (200,400))
            a0_copy = cv2.resize(a0, (200, 400))
            a1_copy = cv2.resize(a1, (200, 400))
            n0_copy = cv2.resize(n0, (200, 400))
            n1_copy = cv2.resize(n1, (200, 400))
            cv2.imshow("spot", s_copy)
            cv2.waitKey(3)
            cv2.imshow("a0", a0_copy)
            cv2.waitKey(3)
            cv2.imshow("a1", a1_copy)
            cv2.waitKey(3)
            cv2.imshow("n0", n0_copy)
            cv2.waitKey(3)
            cv2.imshow("n1", n1_copy)
            cv2.waitKey(3)


        return s, a0, a1, n0, n1

    def detect_image(self, data):
        cropped_img_set_letters = []
        cropped_img_set_numbers = []

        s, a0, a1, n0, n1 = self.cropper(data)
        cropped_img_set_numbers.append(s)
        cropped_img_set_letters.append(a0)
        cropped_img_set_letters.append(a1)
        cropped_img_set_numbers.append(n0)
        cropped_img_set_numbers.append(n1)

        def labelimage(plateID):

            encodingkey = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                           11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                           21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
                           31: '5', 32: '6', 33: '7', 34: '8', 35: '9'}


            label = encodingkey[plateID]
            return label
        predicted = ""


        img = np.asarray(cropped_img_set_numbers[0])
        img_aug = np.expand_dims(img, axis=0)
        img_aug = np.expand_dims(img_aug, axis=3)

        # cv2.imshow("abouttabedetected", img)
        # cv2.waitKey(3)

        with graph1.as_default():
            set_session(sess1)
            predicted_label = self.plate_model_numbers.predict(img_aug)[0]
            # print(predicted_label)
        predicted += labelimage(np.argmax(predicted_label)+26)

        for i in range(0,len(cropped_img_set_letters)):
            img = np.asarray(cropped_img_set_letters[i])
            img_aug = np.expand_dims(img, axis=0)
            img_aug = np.expand_dims(img_aug, axis=3)

            # cv2.imshow("abouttabedetected", img)
            # cv2.waitKey(3)

            with graph1.as_default():
                set_session(sess1)
                predicted_label = self.plate_model_letters.predict(img_aug)[0]
                # print(predicted_label)
            predicted += labelimage(np.argmax(predicted_label))
        for i in range(1, len(cropped_img_set_numbers)):
            img = np.asarray(cropped_img_set_numbers[i])
            img_aug = np.expand_dims(img, axis=0)
            img_aug = np.expand_dims(img_aug, axis=3)

            # cv2.imshow("abouttabedetected", img)
            # cv2.waitKey(3)

            with graph1.as_default():
                set_session(sess1)
                predicted_label = self.plate_model_numbers.predict(img_aug)[0]
                # print(predicted_label)
            predicted += labelimage(np.argmax(predicted_label)+26)


        return predicted

    def most_common_plate(self, number):
        return Counter(self.history[number]).most_common(1)[0]



    def step(self, data):
        processed_image, plate_detected = self.process_image(data)
        if plate_detected:
            unparsed_string = self.detect_image(processed_image)
            print("Guessed: " + unparsed_string)
            #print(unparsed_string[0])


            try:
                self.history[unparsed_string[0]].append(unparsed_string)
                self.process_counter += 1
                temp_string = self.most_common_plate(unparsed_string[0])[0]
                if self.process_counter % 30 == 0:
                    self.process_counter = 0
                    for j in range(0,len(self.history)):
                        print("////////////////////")
                        print("Position:  " + str(j+1))
                        history_j = self.history[str(j+1)]
                        if len(history_j) > 0:
                            print(self.history[str(j+1)][1:])
                            print("The most common plate in this position is: " + self.most_common_plate(str(j+1))[0][1:])
                        else:
                            print("Empty")
                        print("--------------------")
                #print(temp_string)
                self.plate_pub.publish(str(self.TeamID + "," + self.password + ","+temp_string[0]+"," + temp_string[1:]))
                print("Published: " + str(self.TeamID + "," + self.password + "," + temp_string[0] + "," + temp_string[1:]))
                print("      ")
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
