# author：jiangchiying
# date：2019-07-30 19:28
# tool：PyCharm
# Python version：3.7.1

from CNNs.Alexnet import Alexnet_NetWork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2
import os

import matplotlib
matplotlib.use("Agg")

'''
Study resource:  https://keras.io/

A script for training real-time liveness tests.

'''
import matplotlib.pyplot as plt

# Setting the learning rate, batch size and epochs for training model.
Learning_rate = 0.0001
Batch_size = 32
Epochs = 20

Image_size = 224

print("[INFO] loading images...")
# Get the training/testing data set from the image path
Traing_Image_paths = list(paths.list_images('Userdataset'))

# Loading the Traing data from NUAAdata set.
def loading_data(Image_paths):
    # Set the array list to store the  data.
    data = []
    # Set the label array list to store fake or real label
    Label = []
    for image_path in Image_paths:
        # Get the label from the folder name.
        label = image_path.split(os.path.sep)[-2]
        # Read the image from the image_path.
        image = cv2.imread(image_path)
        # Normalize the image into 224X224 size.
        image = cv2.resize(image, (Image_size, Image_size))
        # Store the image into Training data set and label set.
        data.append(image)
        Label.append(label)

    # print(data)
    data = np.array(data, dtype="float")

    print("There are {} images for training/Testing".format(len(data)))
    # labels = 1.0 is fake, 0.1 is real
    le = LabelEncoder()
    labels = le.fit_transform(Label)
    labels = np_utils.to_categorical(labels, 2)
    # print(le.classes_)
    return data, labels



def Training_model(data, labels):

    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.5, random_state=42)

    aug = ImageDataGenerator(rotation_range=30, zoom_range=0.2,
                             width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")


    print("[INFO] Compiling model...")
    opt = Adam(lr=Learning_rate)
    model = Alexnet_NetWork.build(width=Image_size, height=Image_size, depth=3,
                              classes=2)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    model.summary()
    # # train the network
    print("[INFO] Training network for {} epochs...".format(Epochs))
    model.fit_generator(aug.flow(trainX, trainY, batch_size=Batch_size),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // Batch_size,
                            epochs=Epochs)

    # Alexnet = model.fit(TrainX, TrainY, batch_size=Batch_size, epochs=Epochs)

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=Batch_size)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=['fake','real']))

    # save the network to disk
    print("[INFO] serializing network to '{}'...".format('ModelandPickle/Realtimev2.model'))
    model.save('ModelandPickle/Realtimev2.model')


if __name__ == "__main__":

    TrainX, TrainY =loading_data(Traing_Image_paths)
    # Strat Training...
    Training_model(TrainX,TrainY)
