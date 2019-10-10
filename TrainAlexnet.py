# author：jiangchiying
# date：2019-07-25 13:40
# tool：PyCharm
# Python version：3.7.1

from CNNs.Alexnet import Alexnet_NetWork
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


'''
Study resource:  https://keras.io/

A script for training NUAA for Replay-attack dataset by CNN-based livenss tests method.

If you want to change the Training and Testing image path.

Please reset the Traing_Image_paths and Testing_Image_paths parameter.

'''
# Setting the learning rate, batch size and epochs for training model.
Learning_rate = 0.0001
Batch_size = 32
# Epochs = 30
Epochs = 30
Image_size = 224

# Get the training/testing data set from the image path

Traing_Image_paths = list(paths.list_images('NUAAdataset'))
Testing_Image_paths = list(paths.list_images('NUAATestingset'))

# Traing_Image_paths = list(paths.list_images('ReplayAttackTrainset'))
# Testing_Image_paths = list(paths.list_images('ReplayAttackTestset'))

# Loading the Traing data from data set.
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

    print("[INFO] There are {} images for training/Testing....".format(len(data)))
    # labels = 1.0 is fake, 0.1 is real
    le = LabelEncoder()
    labels = le.fit_transform(Label)
    labels = np_utils.to_categorical(labels, 2)
    # print(le.classes_)
    return data, labels



def Training_model(TrainX, TrainY, TestX, TestY):


    '''

    :param TrainX: The Traing image array list used to Train the CNN model.
    :param TrainY: the Label list used to store the image label : Real or fake.
    :param TestX: The Testing image array list used to Train the CNN model.
    :param TestY: the Label list used to store the image label : Real or fake.
    :return:
    '''


    aug = ImageDataGenerator(rotation_range=30, zoom_range=0.2,
                             width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
                             horizontal_flip=True, fill_mode="nearest")


    print("[INFO] Start to compiling the model...")

    # loading the Adam optimizer, which i have compare with the SGD optimizer. however the SGD seen more suitable for the Alexnet model.
    opt = Adam(lr=Learning_rate)

    # opt = SGD(lr=Learning_rate, momentum=0.9, decay=0.001, nesterov=False)


    # Build the Alexnet model by input 224 X 244 pixels image, with 3 color channel RGB.

    model = Alexnet_NetWork.build(width=Image_size, height=Image_size, depth=3,
                              classes=2)


    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    model.summary()
    # Begin to Train the network..

    TrainDataSize = len(TrainX)

    # Compare the keras fit, fit generator method for Training the model.

    print("[INFO] Training network for {} epochs...".format(Epochs))
    Alexnet = model.fit_generator(aug.flow(TrainX, TrainY, batch_size=Batch_size),
                            validation_data=(TestX, TestY), steps_per_epoch=TrainDataSize / Batch_size,
                            epochs=Epochs)

    # Alexnet = model.fit(TrainX,TrainY,batch_size=Batch_size, validation_data=(TestX, TestY), epochs=Epochs)

    # evaluate the network
    print("[INFO] Evaluating network...")
    predictions = model.predict(TestX, batch_size=Batch_size)
    print(classification_report(TestY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=['fake','real']))

    # save the network to disk
    print("[INFO] Serializing network to '{}'...".format('ModelandPickle/AlexnetTest.model'))
    model.save('ModelandPickle/AlexnetTest.model')

    # plot the accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = Epochs
    plt.plot(np.arange(0, N), Alexnet.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), Alexnet.history["val_acc"], label="val_acc")
    plt.title("Training  Accuracy and Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('ResultIMG/AlexnetTest.jpg')

if __name__ == "__main__":

    print("[INFO] Loading images...")
    # Create the TrianX , TrianY for the CNN.
    TrainX, TrainY =loading_data(Traing_Image_paths)

    # Create the Testing data for the CNN.
    TestX, TestY = loading_data(Testing_Image_paths)

    # Strat Training...
    Training_model(TrainX,TrainY,TestX,TestY)
