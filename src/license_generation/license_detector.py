# -*- coding: utf-8 -*-
"""License_detector.ipynb


Original file is located at
    https://colab.research.google.com/drive/1mEHJS_w_xXUQ-X23mkAxTcyvv3wrQOw3

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

https://stackoverflow.com/questions/43382045/keras-realtime-augmentation-adding-noise-and-contrast

"""

# Commented out IPython magic to ensure Python compatibility.
import string
import random
from random import randint
import cv2
import numpy as np
import os
import re
import seaborn as sn
import pandas as pd
import sklearn
import sklearn.metrics

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
# %tensorflow_version 1.14.0
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from keras.preprocessing.image import ImageDataGenerator

# !pip install opencv-contrib-python==4.3.0.38
print(cv2.__version__)




#select number of unique plates to be made
#each plate will be augmented 3 additional times.
#total number of plates generated = NUMBER_OF_PLATES*4
NUMBER_OF_PLATES = 100
SAVEH = 900
SAVEW = SAVEH/3
PATH = os.path.dirname(os.path.realpath(__file__)) + "/"

if not os.path.exists(PATH + "pictures"):
    print("Made directory ./pictures")
    os.makedirs(PATH + "pictures")


#takes an image and apply some bluring and noise to imitate actual photos
def blur_n_noise(img):
    randomint = random.randint(1, 3)


    #gaussian blur
    for i in range(0, randomint):
        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.GaussianBlur(img, (27, 27), 0)
        img = cv2.GaussianBlur(img, (27, 27), 0)
        img = cv2.GaussianBlur(img, (27, 27), 0)
        img = cv2.GaussianBlur(img, (9, 9), 0)

    #motion blur on diagonals
    kernel_size = 2*random.randint(1,9)+1
    if randomint % 2 == 0:
        kernel_r = np.identity(kernel_size)
        kernel_r /= kernel_size

    else:
        kernel_r = np.identity(kernel_size)[::-1]
        kernel_r /= kernel_size

    img = cv2.filter2D(img, -1, kernel_r)

    #noise
    VARIABILITY = 6
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    # cv2.imshow("motion blur", img)
    # cv2.waitKey(3)

    return img


for i in range(0, NUMBER_OF_PLATES):
    print("making new plate: " + str(i))
    path = PATH

    # Pick two random letters
    plate_alpha = ""
    for _ in range(0, 2):
        plate_alpha += (random.choice(string.ascii_uppercase))
    num = randint(0, 99)

    # Pick two random numbers
    plate_num = "{:02d}".format(num)

    # Save plate to file

    # Write plate to image
    blank_plate = cv2.imread(path + 'blank_plate.png')

    # To use monospaced font for the license plate we need to use the PIL
    # package.
    # Convert into a PIL image (this is so we can use the monospaced fonts)
    blank_plate_pil = Image.fromarray(blank_plate)
    # Get a drawing context
    draw = ImageDraw.Draw(blank_plate_pil)
    monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
    draw.text((48, 105), plate_alpha + " " + plate_num, (255, 0, 0), font=monospace)
    # Convert back to OpenCV image and save
    blank_plate = np.array(blank_plate_pil)

    # Create parking spot label
    spotindex = i % 8 + 1
    s = "P" + str(spotindex)
    parking_spot = 255 * np.ones(shape=[600, 600, 3], dtype=np.uint8)
    cv2.putText(parking_spot, s, (30, 450), cv2.FONT_HERSHEY_PLAIN, 28,
                (0, 0, 0), 30, cv2.LINE_AA)
    spot_w_plate = np.concatenate((parking_spot, blank_plate), axis=0)

    # Merge unlabelled images
    unlabelled = np.concatenate((255 * np.ones(shape=[600, 600, 3],
                                               dtype=np.uint8), spot_w_plate), axis=0)

    unlabelled_threshold = unlabelled.copy()
    unlabelled_threshold = cv2.cvtColor(unlabelled_threshold, cv2.COLOR_BGR2GRAY)

    thresh = 70
    dim = (SAVEW, SAVEH)
    unlabelled_threshold = cv2.resize(unlabelled_threshold, dim)
    ret, unlabelled_threshold = cv2.threshold(unlabelled_threshold, thresh, 255, cv2.THRESH_BINARY)


    cv2.imwrite(os.path.join(path + "pictures/",
                             "plate_{}{}_{}_{}.png".format(plate_alpha, plate_num, spotindex, 0)), unlabelled_threshold)

    #grab un-altered plate for keras augmentation
    spot_w_plate = np.asarray([np.array(unlabelled)])


    # Run this to see generated plates.
    #matplot does weird stuff to colours but yellow is white and blue is black
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 3, figsize=(20, 20))
        axes = axes.flatten()
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


    # Create data augmentor from Keras
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.01,
        height_shift_range=0.01,
        brightness_range=None,
        shear_range=5,
        zoom_range=0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=0.4,
        preprocessing_function=blur_n_noise,
        data_format=None,
        validation_split=0.3,
        dtype=None, )

    aug_iter = datagen.flow(spot_w_plate)
    aug_plates = [next(aug_iter)[0].astype(np.uint8) for k in range(3)]
    # Write augmented license plates to file
    for j in range(0, 3):
        aug_plates[j] = cv2.cvtColor(aug_plates[j], cv2.COLOR_RGB2GRAY)
        thresh = 180
        dim = (SAVEW, SAVEH)
        aug_plates[j] = cv2.resize(aug_plates[j], dim)

        aug_plates[j] = cv2.adaptiveThreshold(aug_plates[j], 220, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)
        kernel = np.ones((3, 3), np.uint8)
        # cv2.imshow("aug gen", aug_plates[j])
        # cv2.waitKey(3)
        aug_plates[j] = cv2.morphologyEx(aug_plates[j], cv2.MORPH_OPEN, kernel)
        #aug_plates[j] = cv2.Canny(aug_plates[j], 0, 255)
        # cv2.imshow("thresh gen", aug_plates[j])
        # cv2.waitKey(3)
        cv2.imwrite(os.path.join(path + "pictures/",
                                 "plate_{}{}_{}_{}.png".format(plate_alpha, plate_num, spotindex, j + 1)),
                    aug_plates[j])
print(plate_alpha + " " + plate_num)
#plotImages(aug_plates)

def files_in_folder(folder_path):
    '''
    Returns a list of strings where each entry is a file in the folder_path.
  
    Parameters
    ----------
  
    folder_path : str
       A string to folder for which the file listing is returned.
  
    '''
    files_A = os.listdir(folder_path)
    # print(files_A)
    # The files when listed from Google Drive have a particular format. They are
    # grouped in sets of 4 and have spaces and tabs as delimiters.

    # Split the string listing sets of 4 files by tab and space and remove any 
    # empty splits.
    files_B = [list(filter(None, re.split('\t|\s', files))) for files in files_A]

    # Concatenate all splits into a single sorted list
    files_C = []
    for element in files_B:
        files_C = files_C + element
    files_C.sort()

    return files_C


folder = PATH + "pictures/"
dataset = files_in_folder(folder)
# print(dataset)
print(len(dataset))
generated_dataset_size = len(dataset)

imgset_warped = dataset

X_dataset_letters = []
Y_dataset_letters = []
X_dataset_numbers = []
Y_dataset_numbers = []


np.random.shuffle(imgset_warped)


# parses image and adds labels/images to datasets

def croplicenseimage(plate):
    im = Image.open(folder + plate)
    # w, h = im.size
    img = np.asarray(im)
    w = SAVEW
    h = SAVEH
    dim = (50, 100)
    # cv2.imshow("bw", img)
    # cv2.waitKey(3)

    img = cv2.resize(img, (w, h))
    cv2.imshow("bw", img)
    cv2.waitKey(3)
    # img = cv2.Canny(img, 0, 255)
    s = img[int(h / 2.6):int(h / 1.6), int(w / 2):w]
    a0 = img[int(h / 1.44):int(h / 1.16), 0:int(w / 4.0)]
    a1 = img[int(h / 1.44):int(h / 1.16), int(w / 4.0):int(w / 2.0)]
    n0 = img[int(h / 1.44):int(h / 1.16), int(w / 2.0):int(3 * w / 4.0)]
    n1 = img[int(h / 1.44):int(h / 1.16), int(3 * w / 4.0):w]
    s = cv2.resize(s, dim)
    a0 = cv2.resize(a0, dim)
    a1 = cv2.resize(a1, dim)
    n0 = cv2.resize(n0, dim)
    n1 = cv2.resize(n1, dim)

    _, slabel = labelimage(plate[len(plate) - 7 ] + "_" + str(0) + ".png", number=True)
    _, a0label = labelimage(plate[len(plate) - 12 + 0] + "_" + str(0) + ".png",number=False)
    _, a1label = labelimage(plate[len(plate) - 12 + 1] + "_" + str(1) + ".png",number=False)
    _, n0label = labelimage(plate[len(plate) - 12 + 2] + "_" + str(2) + ".png",number=True)
    _, n1label = labelimage(plate[len(plate) - 12 + 3] + "_" + str(3) + ".png",number=True)
    if (s.shape == (100,50,3)):
        print(plate)
        print(str(s.shape) + " s")
        print(str(a0.shape) + " a0")
    Y_dataset_numbers.append(slabel)
    X_dataset_numbers.append(s)
    Y_dataset_letters.append(a0label)
    X_dataset_letters.append(a0)
    Y_dataset_letters.append(a1label)
    X_dataset_letters.append(a1)
    Y_dataset_numbers.append(n0label)
    X_dataset_numbers.append(n0)
    Y_dataset_numbers.append(n1label)
    X_dataset_numbers.append(n1)



# create one hot key for given cropped image.
# return the 'index' of the number on the plate and return the 'label' that hold the vector of the inputted image.

def labelimage(plateID, number):
    encodingkey = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                   'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                   'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33,
                   '8': 34, '9': 35}
    plate = plateID[0]
    index = plateID[2]
    if number == True:
        label = np.zeros(10)
        label[encodingkey[str(plate)]-26] = 1
    else:
        label = np.zeros(26)
        label[encodingkey[str(plate)]] = 1
    return index, label


for img in imgset_warped[:]:
    croplicenseimage(img)

print(Y_dataset_letters[0])
print(Y_dataset_numbers[0])

X_dataset_letters_norm = [np.asarray(i) / 255.0 for i in X_dataset_letters]
X_dataset_letters_norm = np.asarray(X_dataset_letters_norm)
X_dataset_letters_norm = np.expand_dims(X_dataset_letters_norm, axis=3)
Y_dataset_letters_norm = np.asarray(Y_dataset_letters)

X_dataset_numbers_norm = [np.asarray(i) / 255.0 for i in X_dataset_numbers]
X_dataset_numbers_norm = np.asarray(X_dataset_numbers_norm)
X_dataset_numbers_norm = np.expand_dims(X_dataset_numbers_norm, axis=3)
Y_dataset_numbers_norm = np.asarray(Y_dataset_numbers)

print(X_dataset_numbers_norm.shape)
print(X_dataset_letters_norm.shape)

# for i in range(0, len(X_dataset2)):
#     print(str(np.argmax(Y_dataset2[i])) )
#     cv2.imshow("mapping?" ,X_dataset[i])
#     cv2.waitKey(3)
#
#     plt.imshow(X_dataset[i])
#     plt.figure("a")
#     plt.show()

VALIDATION_SPLIT = 0.2


def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

# numberNet
# numberNet
# numberNet
conv_model_numbers = models.Sequential()
conv_model_numbers.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 50, 1)))
conv_model_numbers.add(layers.MaxPooling2D((2, 2)))
conv_model_numbers.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model_numbers.add(layers.MaxPooling2D((2, 2)))
conv_model_numbers.add(layers.Flatten())
conv_model_numbers.add(layers.Dropout(0.5))
conv_model_numbers.add(layers.Dense(512, activation='relu'))
conv_model_numbers.add(layers.Dense(10, activation='softmax'))

LEARNING_RATE = 1e-5
conv_model_numbers.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])
conv_model_numbers.summary()

EPOCHS = 20
BATCHES = 60
reset_weights(conv_model_numbers)  # do if ya want
history_conv_numbers = conv_model_numbers.fit(X_dataset_numbers_norm, Y_dataset_numbers_norm, validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
                              batch_size=BATCHES)

plt.plot(history_conv_numbers.history['loss'])
plt.plot(history_conv_numbers.history['val_loss'])
plt.title('number model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv_numbers.history['acc'])
plt.plot(history_conv_numbers.history['val_acc'])
plt.title('number model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

#letterNet
conv_model_letters = models.Sequential()
conv_model_letters.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 50, 1)))
conv_model_letters.add(layers.MaxPooling2D((2, 2)))
conv_model_letters.add(layers.Conv2D(64, (3, 3), activation='relu'))
conv_model_letters.add(layers.MaxPooling2D((2, 2)))
conv_model_letters.add(layers.Flatten())
conv_model_letters.add(layers.Dropout(0.5))
conv_model_letters.add(layers.Dense(512, activation='relu'))
conv_model_letters.add(layers.Dense(26, activation='softmax'))

LEARNING_RATE = 1e-5
conv_model_letters.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])
conv_model_letters.summary()

EPOCHS = 30
BATCHES = 60
reset_weights(conv_model_letters)  # do if ya want
history_conv_letters = conv_model_letters.fit(X_dataset_letters_norm, Y_dataset_letters_norm, validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
                              batch_size=BATCHES)

plt.plot(history_conv_letters.history['loss'])
plt.plot(history_conv_letters.history['val_loss'])
plt.title('letter model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv_letters.history['acc'])
plt.plot(history_conv_letters.history['val_acc'])
plt.title('letter model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()
#
# def displayImage(index):
#     img = X_dataset_numbers_norm[index]
#     print(img.shape)
#     img_aug = np.expand_dims(img, axis=0)
#     #img_aug = np.expand_dims(img, axis=3)
#     y_predict = conv_model_numbers.predict(img_aug)[0]
#     img_show = img[:,:,0]
#     plt.imshow(img_show)
#     plt.show()
#     encodingkey = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
#                    'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
#                    'X': 23, 'Y': 24, 'Z': 25, '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33,
#                    '8': 34, '9': 35}
#     inv_map = {v: k for k, v in encodingkey.items()}
#     caption = np.argmax(y_predict)
#     plt.title(str(caption))


# interact(displayImage, 
#         index=ipywidgets.IntSlider(min=0, max=X_dataset_orig.shape[0],
# #                                    step=1, value=10))
# for i in range(10, 20):
#     plt.figure(i)
#     displayImage(i)


def predictimage_numbers():
    y_predict = []

    for index in range(0, len(Y_dataset_numbers_norm)):
        img = X_dataset_numbers_norm[index]
        img_aug = np.expand_dims(img, axis=0)
        y_predict.append(np.argmax(conv_model_numbers.predict(img_aug)[0]))
    return y_predict

def predictimage_letters():
    y_predict = []

    for index in range(0, len(Y_dataset_letters_norm)):
        img = X_dataset_letters_norm[index]
        img_aug = np.expand_dims(img, axis=0)
        y_predict.append(np.argmax(conv_model_letters.predict(img_aug)[0]))
    return y_predict




y_predict_numbers = np.round(predictimage_numbers(), 1)
yreal_numbers = []
y_predict_letters = np.round(predictimage_letters(), 1)
yreal_letters = []

for i in range(0, len(Y_dataset_numbers_norm)):
    yreal_numbers.append(np.argmax(Y_dataset_numbers_norm[i]))
for i in range(0, len(Y_dataset_letters_norm)):
    yreal_letters.append(np.argmax(Y_dataset_letters_norm[i]))

conv_model_numbers.save(path + "number_detection_CNN")
conv_model_letters.save(path + "letter_detection_CNN")

conf = sklearn.metrics.confusion_matrix(yreal_numbers, y_predict_numbers)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()
df_cm = pd.DataFrame(conf, index=[i for i in "0123456789"],
                     columns=[i for i in "0123456789"])
plt.figure(figsize=(15, 9))
sn.heatmap(df_cm, annot=True)
plt.show()

conf = sklearn.metrics.confusion_matrix(yreal_letters, y_predict_letters)
plt.imshow(conf, cmap='binary', interpolation='None')
plt.show()
df_cm = pd.DataFrame(conf, index=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                     columns=[i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
plt.figure(figsize=(15, 9))
sn.heatmap(df_cm, annot=True)
plt.show()