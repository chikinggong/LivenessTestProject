# author：jiangchiying
# date：2019-07-22 18:56
# tool：PyCharm
# Python version：3.7.1

from sklearn.svm import SVC
import pickle

'''

A script for training SVM classifier 


'''

def TrainSVM():

    print('Loading the training set from SVM..')
    pickle_in = open('ModelandPickle/LBPTestingXRP.pickle', 'rb')
    LBPTrainX = pickle.load(pickle_in)

    print(len(LBPTrainX))
    # print(TrainingData)


    pickle_in = open('ModelandPickle/LBPTestingYRP.pickle', 'rb')
    LBPTrainY = pickle.load(pickle_in)

    # print(Label)

    print('Loading the Testing set from SVM..')
    pickle_in = open('ModelandPickle/LBPTrainXRP.pickle', 'rb')
    LBPTestingX = pickle.load(pickle_in)

    print(len(LBPTestingX))

    pickle_in = open('ModelandPickle/LBPTrainYRP.pickle', 'rb')
    LBPTestingY = pickle.load(pickle_in)

    # Crete the model

    SVM = SVC(C=1,kernel='rbf',gamma=20,decision_function_shape='ovr', probability=True)
    # SVM.fit(TrainingData, Label)
    SVM.fit(LBPTrainX,LBPTrainY)

    # Save the model as the pickle file.

    s = pickle.dumps(SVM)

    # Saving the model as pickle file.
    f = open('Modelandpickle/LBPSVMtest.pickle','wb+')

    f.write(s)

    f.close()

    print("Save model succuess..")

    print(" Evaluate the model....")

    print("Traing Set Accuracy:",SVM.score(LBPTrainX,LBPTrainY))

    print("Testing Set Accuracy:",SVM.score(LBPTestingX,LBPTestingY))


if __name__ == "__main__":

    # Train SVM
    TrainSVM()
