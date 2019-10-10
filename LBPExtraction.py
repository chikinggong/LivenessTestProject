# author：jiangchiying
# date：2019-07-16 13:11
# tool：PyCharm
# Python version：3.7.1
from LBPLiveness.LBPuniform import *
import os
from imutils import paths
import time
import pickle


# Training_IMG_PATHS = list(paths.list_images('NUAAdataset'))
# Testing_IMG_PATHS = list(paths.list_images('NUAATestingset'))

Training_IMG_PATHS = list(paths.list_images('ReplayAttackTrainset'))
Testing_IMG_PATHS = list(paths.list_images('ReplayAttackTestset'))
lbp = LBPtest()

# Create the training/testing data set by using LBP and then convert into hist
def Create_LBP_data(IMG_PATHS):
    '''

    Create data set

    Args:
            IMG_PATHS : input the normalized image.

     Returns:

            An array list of the histogram and label data.
    '''
    Data = []
    Label = []
    for image_path in IMG_PATHS:

        # extract the class label from the filename, load the image
        label = image_path.split(os.path.sep)[-2]

        # Image Pre Processing
        image_array = lbp.ImagePre_Proccessing(image_path)
        # Get the LBP Uniform from the image.
        uniform_array = lbp.lbp_uniform(image_array)
        # Calculate the histogram from the lbp feature.
        hist = lbp.Calculate_hist(uniform_array)
        # Store the histogram feature into the data array list.
        Data.append(hist)
        # Store the label feature into the Label array list.
        Label.append(label)

    return Data, Label

if __name__ == "__main__":

    print("[INFO] Start to create the dataset..... It make cost 10 mins")

    Starttime = time.time()

    TrainX, TrainY =Create_LBP_data(Training_IMG_PATHS)

    TestX, TestY = Create_LBP_data(Testing_IMG_PATHS)

    # Create the Pickle file to store the TrainX, TrainY, TestX, TestY for the SVM.

    # print(TrainX)
    LBPTrainX = []
    LBPTrainY = []
    for feature in TrainX:
        LBPTrainX.append(feature)

    for label in TrainY:
        LBPTrainY.append(label)
    #     Store the training set as pickle file.
    pickle_out = open('ModelandPickle/LBPTrainXRP.pickle', 'wb')
    pickle.dump(LBPTrainX, pickle_out)
    pickle_out.close()
    #     Store the training label  as pickle file.
    pickle_out = open('ModelandPickle/LBPTrainYRP.pickle', 'wb')
    pickle.dump(LBPTrainY, pickle_out)
    pickle_out.close()

    LBPTestingX = []
    LBPTestingY = []
    for feature in TestX:
        LBPTestingX.append(feature)

    for label in TestY:
        LBPTestingY.append(label)
    #     Store the testing set as pickle file.
    pickle_out = open('ModelandPickle/LBPTestingXRP.pickle', 'wb')
    pickle.dump(LBPTestingX, pickle_out)
    pickle_out.close()
    #     Store the testing label as pickle file.
    pickle_out = open('ModelandPickle/LBPTestingYRP.pickle', 'wb')
    pickle.dump(LBPTestingY, pickle_out)
    pickle_out.close()

    print("Create LBP Train/ Test set success!!")
    Endtime = time.time()

    Costtime = Endtime - Starttime
    # calculate the cost-time
    print("The Cost time is ", Costtime)





