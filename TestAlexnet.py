# author：jiangchiying
# date：2019-07-16 16:17
# tool：PyCharm
# Python version：3.7.1

# Import some necessary library
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import logging
import pickle
import cv2
import os
from imutils import paths
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
logging.getLogger('tensorflow').disabled = True

'''
A Script for testing the CNN-based liveness tests in NUAA or Replay Attack.


'''

# Loading the model which you want to test.

model = load_model("ModelandPickle/Alexnetv1.model")

# Loading the testing set from the image path.

IMG_PATHS=list(paths.list_images('ReplayAttackTestset'))

RealLabel = []

image_label=['fake', 'real']

y_score=[]

# Record the real label of the image.
for image_path in IMG_PATHS:
    label = image_path.split(os.path.sep)[-2]
    # print(label)

    # update the data and labels lists, respectively
    RealLabel.append(label)

# record the test label of the testing image.
Testingdata = []
for image_path in IMG_PATHS:
    testimage = cv2.imread(image_path)
    test_image = cv2.resize(testimage, (224, 224))
    test_image = test_image.astype("float")
    test_image = img_to_array(test_image)
    face = np.expand_dims(test_image, axis=0)
    preds = model.predict(face)[0]
    j = np.argmax(preds)
    result = int(j)
    label = image_label[result]
    Testingdata.append(label)
    y_score.append(preds[j])

# print(Testingdata)

y_true = RealLabel
y_pred = Testingdata

TP,FN,FP,TN=confusion_matrix(y_true, y_pred).ravel()
FPR, TPR, thresholds= roc_curve(y_true, y_score,pos_label='fake')

# TP,FN,FP,TN =c_matrix
# print("Confusion Matrix:")
# print(c_matrix)

print("True Positive: ",TP)
print("False Negative: ",FN)
print("False Positive: ",FP)
print("True Negative: ",TN)

Accary = ((TP+TN)/(TP+FN+FP+TN))*100

print('Accuracy is:', Accary)

FAR = FP/(FP+TN)

FRR = FN/(TP+FN)

HTER = ((FRR+FAR)/2)*100

print('Half Total Error Rate is: ', HTER)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(FPR, TPR, label='Alexnet')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()