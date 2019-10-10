# author：jiangchiying
# date：2019-07-01 18:47
# tool：PyCharm
# Python version：3.7.1


# Import some necessary library
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import os
import pickle
import dlib

import logging
logging.getLogger('tensorflow').disabled = True


# Loading the haarcascade from the FaceDetector folder.
OSSeparator = os.path.sep
detectorDir = 'FaceDetector' + OSSeparator + 'haarcascade_frontalface_default.xml'

# Loading the face detector
print("[INFO] loading face detector...")
faceCascade = cv2.CascadeClassifier(detectorDir)

# load  model and label encoder from disk
print("[INFO] loading the cnn model and pickle...")
model = load_model("ModelandPickle/Realtimev2.model")
image_label=['fake', 'real']
# Detect method
# frame is the variable capture from the web camera.
def detect(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in face:
        # print(x, y, w, h)
        color = (0, 0, 255)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        # Capture the  face from the frame and then test it
        test_image = frame[y + 10: y + h - 10, x + 10: x + w - 10]
        test_image = cv2.resize(test_image, (224, 224))
        test_image = test_image.astype("float")
        test_image = img_to_array(test_image)
        face = np.expand_dims(test_image, axis=0)
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        # label = le.classes_[j]
        result = int(j)
        label = image_label[result]
        if(label=='real'):
            label = "{}: {:.3f}".format(label, preds[j])
            cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
        elif(label=='fake'):
            label = "{}: {:.3f}".format(label, preds[j])
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('/Users/jiangchiying/Downloads/replayattack/test/real/client115_session01_webcam_authenticate_controlled_2.mov')
while True:
    # Capture the frame from the web camera.
    ret, frame = cap.read()

    detect(frame)
    frame = cv2.putText(frame, 'Press Q to exit', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
    # Display the real time frame.
    cv2.imshow('Face Detection', frame)

    # If user type Q from the keyboard the loop could be break.
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



# Release the resource and destroy all windows.
cap.release()
cv2.destroyAllWindows()