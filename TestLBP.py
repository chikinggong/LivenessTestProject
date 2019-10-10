# author：jiangchiying
# date：2019-08-11 14:11
# tool：PyCharm
# Python version：3.7.1
import pickle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

'''
A Script for testing the LBP-based method 

the pickle_in can be reset according to what kind of training/testing you want to set.

For example, the LBPTestingXRP is ReplayAttack testing set. LBPTestingX3 is NUAA testing set.
LBPSVM3 is NUAA LBP model
LBOSVMRP is Replay-Attack model.


'''

def TestLBP():

    print('Loading the Testing set from SVM..')
    pickle_in = open('ModelandPickle/LBPTestingXRP.pickle', 'rb')
    # pickle_in = open('ModelandPickle/LBPTestingX3.pickle', 'rb')
    LBPTestingX = pickle.load(pickle_in)

    pickle_in = open('ModelandPickle/LBPTestingYRP.pickle', 'rb')
    # pickle_in = open('ModelandPickle/LBPTestingX3.pickle', 'rb')
    LBPTestingY = pickle.load(pickle_in)

    # Loading the testing model from ModelandPickle file.
    file = open('ModelandPickle/LBPSVM3.pickle','rb')

    s = file.read()
    model = pickle.loads(s)

    score = model.score(LBPTestingX,LBPTestingY)

    # Convert the score into precentage
    score = score *100

    y_pred = model.predict(LBPTestingX)

    TP, FN, FP, TN = confusion_matrix(LBPTestingY, y_pred).ravel()

    FAR = FP / (FP + TN)

    FRR = FN / (TP + FN)

    HTER = ((FRR + FAR) / 2) * 100


    print("True Positive: ", TP)
    print("False Negative: ", FN)
    print("False Positive: ", FP)
    print("True Negative: ", TN)

    print("The Accuracy of LBP + SVM is ",score)

    print('Half Total Error Rate is: ', HTER)



if __name__ == "__main__":

    # Test LBP + SVM
    TestLBP()