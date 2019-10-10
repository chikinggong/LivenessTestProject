# author：jiangchiying
# date：2019-07-09 11:39
# tool：PyCharm
# Python version：3.7.1

# Import the necessary library
import cv2
import os

'''

The class of face detector

'''

# Loading the haarcascade from the FaceDetector folder.
OSSeparator = os.path.sep
detectorDir = 'FaceDetector' + OSSeparator + 'haarcascade_frontalface_default.xml'



class Detector(object):
    def __init__(self, data_filename=detectorDir):
        super(Detector, self).__init__()
        self.face_cascade = cv2.CascadeClassifier(data_filename)

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        return face
