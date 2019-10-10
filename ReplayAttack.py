# author：jiangchiying
# date：2019-08-02 18:23
# tool：PyCharm
# Python version：3.7.1

import cv2
import os
from Detector import Detector
import shutil

'''
A Script for dealing with the Replay-Attack data set.

One video has 374 frame if you want to capture more face image for train/test. 

Please set the gap in Getface() method.
'''

Dir = 'ReplayAttackTrainset'

Dir2 = 'ReplayAttackTestset'

# Set the Train Real dir

TrainRealDir = '/Users/jiangchiying/Downloads/replayattack/train/real'

TrainRealStorDir = 'ReplayAttackTrainset/real'


# Set the Train fake dir
TrainFakeDir = '/Users/jiangchiying/Downloads/replayattack/train/attack/fixed'

TrainfakeStorDir = 'ReplayAttackTrainset/fake'


# Set the Test real dir
TestRealDir = '/Users/jiangchiying/Downloads/replayattack/test/real'
TestRealStorDir = 'ReplayAttackTestset/real'


# Set the test fake dir
TestFakeDir = '/Users/jiangchiying/Downloads/replayattack/test/attack/fixed'
TestfakeStorDir = 'ReplayAttackTestset/fake'




IMG_SIZE = 224



def Get_file_name(file_dir):
    Filename = []
    for root, dirs, files in os.walk(file_dir):
        Filename.append(files)
    return Filename

# A method to count the total number of files
def count_files(path_name):
    import os
    ls = os.listdir(path_name)
    count = 0
    for i in ls:
        if os.path.isfile(os.path.join(path_name, i)):
            count += 1
    return count

def delete_files(path_name):
    if not os.listdir(path_name):
        print("Empty file.. ")
        return
    os.chdir(path_name)
    fileList = list(os.listdir())
    for file in fileList:
        if os.path.isfile(file):
            os.remove(file)
            print("delete successfully")
        else:
            shutil.rmtree(file)

# A method for create real and fake folder.
def Createfolder(path):
    file_name = ['/real', '/fake']
    for name in file_name:
        os.mkdir(path + name)

# Get the face image from the video.
def Getface(Filename,Video_Dir,FramNum,Save_Dir):
    count = 0
    for filenames in Filename:
        for filename in filenames:
            Videoname= Video_Dir+'/'+filename

            vc = cv2.VideoCapture(Videoname)
            flag, frame = vc.read()
            # Calculating total frames
            frame_count = 0
            while (flag):
                ret, frame = vc.read()
                if ret is False:
                    break
                frame_count = frame_count + 1
            vc.release()

            gap = frame_count // FramNum
            c = 1

            vc = cv2.VideoCapture(Videoname)
            flag, frame = vc.read()
            while (flag):
                flag, frame = vc.read()
                if (flag == 0):
                    break
                if (c % gap == 0):
                    facedector = Detector()
                    # print(self.image)
                    face = facedector.detect_face(frame)
                    if len(face) > 0:
                        for (x, y, w, h) in face:
                            Crop_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            Crop_image = Crop_face[y: y + h, x: x + w]
                            Crop_image = cv2.cvtColor(Crop_image, cv2.COLOR_BGR2RGB)
                            Crop_image = cv2.resize(Crop_image,(IMG_SIZE, IMG_SIZE))
                            cv2.imwrite(Save_Dir + '/' + str(count) + '.jpg', Crop_image)
                            print("image saving....")
                            count = count + 1
                c = c + 1
            cv2.waitKey(1)




if __name__ == "__main__":

    # A method to Clear the data....
    # delete_files(Dir)
    # delete_files(Dir2)

    Createfolder(Dir)
    Createfolder(Dir2)
    # Get the Replay attack video name from the folder.
    TrainRealFilename = Get_file_name(TrainRealDir)
    print('[INFO] Start to capture the users face into the ReplayAttackTrainset')
    Getface(TrainRealFilename,TrainRealDir, 40,TrainRealStorDir)

    TrainFakeFilename = Get_file_name(TrainFakeDir)
    print('[INFO] Start to capture the fake face into the ReplayAttackTrainset')
    Getface(TrainFakeFilename,TrainFakeDir, 20,TrainfakeStorDir)


    # Create the testing set
    TestRealFilename = Get_file_name(TestRealDir)
    print('[INFO] Start to capture the users face into the ReplayAttackTestset')
    Getface(TestRealFilename, TestRealDir, 40, TestRealStorDir)

    TestFakeFilename = Get_file_name(TestFakeDir)
    print('[INFO] Start to capture the fake face into the ReplayAttackTestnset')
    Getface(TestFakeFilename, TestFakeDir, 20, TestfakeStorDir)
