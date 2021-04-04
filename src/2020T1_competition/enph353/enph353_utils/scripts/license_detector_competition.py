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
        self.path = "/home/fizzer/ros_ws/src/license_generation/"
        self.history = {'1' : [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8' : []}
        # initialize game
        rospy.init_node('license_publisher')
        self.plate_model = tensorflow.keras.models.load_model(self.path + "detection_CNN")
        self.process_counter = 0



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
        #cv2.imshow("raw", cv_image)
        #cv2.waitKey(3)
        #img_orig.shape

        row, col, plane = img_orig.shape
        #plt.imshow(img_orig)
        #plt.figure("whole")
        #plt.show()
        # crop image
        img = img_orig[int(row * 2 / 5):int(row * 7 / 8), :, :].copy()
        img_crop = img.copy()

        # Threshold in hsv
        # THESE ARE REALLY GOOD DONT CHANGE
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        lower_spot_grey = np.array([0, 0, 90])
        upper_spot_grey = np.array([20, 30, 210])

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
        #cv2.imshow("hsv altercation", result)
        #cv2.waitKey(3)
       # plt.imshow(result)
        #plt.figure("whole")
        #plt.show()
        # mask and threshold
        result_grey = result[:, :, 2]
        thresh = 40
        ret, result_binary = cv2.threshold(result_grey, thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        result_binary = cv2.morphologyEx(result_binary, cv2.MORPH_OPEN, kernel)
        result_binary = cv2.morphologyEx(result_binary, cv2.MORPH_CLOSE, kernel)
        #plt.imshow(result_binary)
        #plt.figure("masked")
        #plt.show()

        # edge detext
        edged = cv2.Canny(result_binary, 20, 150)
        #plt.figure('edges')
        #plt.title('edges')
        #plt.imshow(edged)
        #plt.show()

        # compute contours
        _, contours, _= cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        if len(contours) > 0:
            contours_poly = [None] * len(contours)
            #boundRect = [None] * len(contours)

            contours_poly[0] = cv2.approxPolyDP(contours[0], 20, True)
            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            # print(contours_poly[0])
            # print(boundRect[0])

            # make sure its square-ish and nothing else!
            if len(contours_poly[0]) == 4:
                # print(boundRect[0])

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

                if (bx - tx) * (by - ty) > 50 * 50:
                    # compute the difference between the points -- the top-right
                    # will have the minumum difference and the bottom-left will
                    # have the maximum difference
                    diff = np.diff(rect_pts, axis=1)
                    rect[1] = rect_pts[np.argmin(diff)]
                    rect[3] = rect_pts[np.argmax(diff)]
                    targetwidth = 600
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
                    #img_crop = cv2.drawContours(img_crop, [contours_poly[0]], 0, (0, 0, 255), 2)
                    self.process_counter += 1
                   # framenumber = self.process_counter % 30 +1
                    #frameinfo = cv2.putText(dst, str(framenumber), (30, 200), cv2.FONT_HERSHEY_PLAIN, 28,
                     #           (0, 0, 0), 10, cv2.LINE_AA)
                    #cv2.imshow("plate", frameinfo)
                    #cv2.waitKey(3)
                    # some proof that its working
                    #plt.imshow(img_crop)
                    #plt.figure("whfdsaole")
                    #plt.show()
                    #plt.imshow(dst)
                    #plt.figure("whfdsaole")
                    #plt.show()

                    # detected is true when a
                    detected = True
        processed_image = dst
        plate_detected = detected
        return processed_image, plate_detected
        # processed_image = []
        # HOMOGRAPHY_POINTS = 15
        # # cv2.imshow("raw", cv_image)
        # # cv2.waitKey(3)
        # cv_image_copy = cv_image.copy()
        # sift = cv2.xfeatures2d.SIFT_create()
        #
        # img = cv2.imread(self.path + "aplate_blank_flat.png")
        #
        # # #plt.imshow(img)
        # targetheight = img.shape[0]
        # targetwidth = img.shape[1]
        #
        # kp_image, desc_image = sift.detectAndCompute(img, None)
        # # Feature matching
        # index_params = dict(algorithm=0, trees=5)
        # search_params = dict()
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        #
        # frame = cv_image_copy
        #
        # grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # trainimage
        # kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
        # matches = flann.knnMatch(desc_image, desc_grayframe, k=2)
        # good_points = []
        # for m, n in matches:
        #     if m.distance < 0.68 * n.distance:
        #         good_points.append(m)
        #
        # if len(good_points) > HOMOGRAPHY_POINTS:
        #     query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        #     train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        #     matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        #     matches_mask = mask.ravel().tolist()
        #
        #     # Perspective transform
        #     h, w, _ = img.shape
        #     print("SIFT IMAGE: " + str(img.shape))
        #     pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        #     dst = cv2.perspectiveTransform(pts, matrix)
        #     # print(np.int32(dst))
        #     homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
        #     # plt.figure(str(i) + "matched")
        #     # plt.imshow(frame)
        #     # plt.show()
        #
        #     # If plate is identified, stretch it and save it.
        #     rect_pts = np.int32(dst).reshape(4, 2)
        #     rect = np.zeros((4, 2), dtype="float32")
        #     # the top-left point has the smallest sum whereas the
        #     # bottom-right has the largest sum
        #     s = rect_pts.sum(axis=1)
        #     rect[0] = rect_pts[np.argmin(s)]
        #     rect[2] = rect_pts[np.argmax(s)]
        #     # compute the difference between the points -- the top-right
        #     # will have the minumum difference and the bottom-left will
        #     # have the maximum difference
        #     diff = np.diff(rect_pts, axis=1)
        #     rect[1] = rect_pts[np.argmin(diff)]
        #     rect[3] = rect_pts[np.argmax(diff)]
        #
        #     # now that we have our rectangle of points, let's compute
        #     # the width of our new image
        #     (tl, tr, br, bl) = rect
        #     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        #     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        #     # ...and now for the height of our new image
        #     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        #     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        #     # take the maximum of the width and height values to reach
        #     # our final dimensions
        #     # construct our destination points which will be used to
        #     # map the screen to a top-down, "birds eye" view
        #     dst = np.array([
        #         [0, 0],
        #         [targetwidth - 1, 0],
        #         [targetwidth - 1, targetheight - 1],
        #         [0, targetheight - 1]], dtype="float32")
        #     # calculate the perspective transform matrix and warp
        #     # the perspective to grab the screen
        #     M = cv2.getPerspectiveTransform(rect, dst)
        #     warp = cv2.warpPerspective(frame, M, (targetwidth, targetheight))
        #     cv2.imshow("detected", warp)
        #     cv2.waitKey(3)
        #     # plt.figure("warped")
        #     # plt.imshow(warp)
        #     # plt.show()
        #     #print(warp.shape)
        #     processed_image = warp
        #     plate_detected = True
        # else:
        #
        #     framenotmatched = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
        #     cv2.imshow("NOT!!! detected", framenotmatched)
        #     cv2.waitKey(3)
        #     # plt.figure("notmatched")
        #     # plt.imshow(framenotmatched)
        #     # plt.show()

    def cropper(self, img):
        w = 200
        h = 400
        dim = (50, 100)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("bw", img)
        # cv2.waitKey(3)
        thresh = 77
        ret, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        # cv2.imshow("thresh", img)
        # cv2.waitKey(3)
        img = cv2.resize(img, (w, h))
        #img = cv2.Canny(img, 0, 255)
        cv2.imwrite(os.path.join(self.path + "warpedpictures/",
                                 "plate{}.png".format(self.process_counter)), img)
        s = img[int(h / 3.0):int(2.0 * h / 3.0), int(w / 2):w]
        a0 = img[int(2.0 * h / 3.0):int(8 * h / 9.0), 0:int(w / 4.0)]
        a1 = img[int(2 * h / 3.0):int(8 * h / 9.0), int(w / 4.0):int(w / 2.0)]
        n0 = img[int(2 * h / 3.0):int(8 * h / 9.0), int(w / 2.0):int(3 * w / 4.0)]
        n1 = img[int(2 * h / 3.0):int(8 * h / 9.0), int(3 * w / 4.0):w]
        s = cv2.resize(s, dim)
        a0 = cv2.resize(a0, dim)
        a1 = cv2.resize(a1, dim)
        n0 = cv2.resize(n0, dim)
        n1 = cv2.resize(n1, dim)

        # s = cv2.Canny(s, 0, 255)
        # a0 = cv2.Canny(a0, 0, 255)
        # a1 = cv2.Canny(a1, 0, 255)
        # n0 = cv2.Canny(n0, 0, 255)
        # # n1 = cv2.Canny(n1, 0, 255)
        # cv2.imshow("s", s)
        # cv2.waitKey(3)
        # cv2.imshow("a0", a0)
        # cv2.waitKey(3)
        # cv2.imshow("a1", a1)
        # cv2.waitKey(3)
        # cv2.imshow("n0", n0)
        # cv2.waitKey(3)
        # cv2.imshow("n1", n1)
        # cv2.waitKey(3)



        return s, a0, a1, n0, n1

    def detect_image(self, data):
        cropped_img_set = []

        s, a0, a1, n0, n1 = self.cropper(data)
        cropped_img_set.append(s)
        cropped_img_set.append(a0)
        cropped_img_set.append(a1)
        cropped_img_set.append(n0)
        cropped_img_set.append(n1)

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
            img_aug = np.expand_dims(img_aug, axis=3)

            # cv2.imshow("abouttabedetected", img)
            # cv2.waitKey(3)

            with graph1.as_default():
                set_session(sess1)
                predicted_label = self.plate_model.predict(img_aug)[0]
                # print(predicted_label)
            predicted += labelimage(np.argmax(predicted_label))


        return predicted

    def most_common_plate(self, number):
        return Counter(self.history[number]).most_common(1)[0]



    def step(self, data):
        processed_image, plate_detected = self.process_image(data)
        if plate_detected:
            unparsed_string = self.detect_image(processed_image)
            #print(unparsed_string)
            #print(unparsed_string[0])


            try:
                self.history[unparsed_string[0]].append(unparsed_string)
                temp_string = self.most_common_plate(unparsed_string[0])[0]
                #print(temp_string)
                self.plate_pub.publish(str(self.TeamID + "," + self.password + ","+temp_string[0]+"," + temp_string[1:]))
                print("published: " + str(self.TeamID + "," + self.password + "," + temp_string[0] + "," + temp_string[1:]))
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
