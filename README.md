# LivenessTestProject

PROJECT TITLE: Liveness tests for face recognition

PURPOSE OF PROJECT: Develop a liveness test system with a LBP-based method and CNN-based method. And also, a real-time liveness test was created by CNN method. This is a Durham University master degree final project created by Chiying Jiang.
VERSION or DATE: 19/8/19
AUTHORS: Chiying Jiang
HOW TO START THIS PROJECT: Run the GUI.py in this project can loading the main interface.
And also, when teåsting the image in this system, the images could find in the raw folder in Testing/Evaluation data folder.


Notes: Before running the project make sure, you have install the Dependencies in your IDE environment.

DEPENDENCIES:

Python 3.5
Tkinter
OpenCV
Numpy
Keras
Tensorflow
Sklearn

FOLDER INTRODUCTION:

There are many folders are used to training and testing. In this part, it will explain what is the purpose of every Python file.

CNNs folder: this folder is used for storing the class of convolutional nerual network.
In this file, you can find a AlexNet and a VGG-11 network. But the VGG-11 was abandoned for this project because the testing result is not good enough.

Face Detector: this is the folder to store the face detector model.

LBPLiveness folder: this folder is store the class of uniformLBP, which is used for extra the feature from face image. The Detail comment of uniformLBP can check in the Python file.

ModelandPickle: storing the trained model, which would be the .pickle file for LBP liveness tests and .model for CNN liveness tests. 

NUAAdataset/NUAATestingset: the folder for storing the training set and testing set.

ReplayAttackTrainset/ReplayAttackTestset: the folder for storing the training set and testing set.

ResultIMG: the folder for storing the CNN model loss and accuracy image.

Userdataset: the folder for storing the custom dataset for real-time detection.




PYTHON FILE INTRODUCTION AND USER INSTRUCTION:

Running the GUI.py can start to run the system.

If you want to train and test the LBP and CNN liveness detection again, you can follow the instruction in the below.

The first step is to run both NUAADdatapreparation and ReplayAttack python file. It could be a long time for loading the image.
This part is finished the image pre-processing also.

NUAADatapreparation.py: this file is to divide the NUAA dataset into fake/real training/testing set.

ReplayAttack.py: this file is to divide the Replay-Attack dataset into fake/real training/testing set.


The second step is to Training the LBP and CNN liveness tests methods.

LBPExtraction.py: this python file is aimed to capture the feature from a training set and testing set. This should be run before the TrainSVM.py

TrainSVM.py: once the LBPExtraction.py was finished, the user can train their own model to predict image.

TrainingAlexnet.py: this script is aimed at training the CNN-based liveness tests model.

TrainRealtime.py: this script is to train the real-time liveness detection.


After training the new model, you can change the model path in the GUI.py for testing your new model.

The third step is to testing the trained model.

TestLBP.py: can be used for test the LBP performance. 

TestAlexnet.py: can be used for testing the CNN performance.


Other useful python files:

CropFaceFromCam.py: this script is used to capture the user face for training custom dataset. The command of running this script is: python CropFaceFromCam.py --output Userdataset/(real or fake)

Detector.py: this is the class of Face detector, using for capture the face from image.

