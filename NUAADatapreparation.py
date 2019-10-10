# author：jiangchiying
# date：2019-07-09 11:36
# tool：PyCharm
# Python version：3.7.1

'''
This is the script to load the Training data set from the NUAA data set folder
User should correct entre the path and fake_path address.
'''
# Import the necessary library
import cv2
import os
import sys
import shutil


# Defend the Client image path and fake Image path.
path = '/Users/jiangchiying/Desktop/Detectedface/ClientFace'
fake_path = '/Users/jiangchiying/Desktop/Detectedface/ImposterFace'

# Select how many category use to create the data set

# CATEGORIES = ["0001", "0002", "0003", "0004","0005","0006","0007"]
Traing_Real_CATEGORIES = ["0001", "0002", "0003", "0004","0005","0006","0007"]

Traing_Fake_CATEGORIES = ["0001", "0002", "0003", "0004","0005"]

Test_Real_CATEGORIES = ["0008", "0009", "0010","0011","0012","0014","0015"]

Test_Fake_CATEGORIES = ["0006","0007","0008", "0009", "0010","0011","0012","0014","0015"]

Dir = 'NUAAdataset'
Dir2 = 'NUAATestingset'

# Setting the store image path
StoreDir = 'NUAAdataset/real'
Fake_StoreDir = 'NUAAdataset/fake'

Test_StoreDir = 'NUAATestingset/real'
Test_Fake_StoreDir = 'NUAATestingset/fake'

# Resize the file into 224 X 224 pixel
IMG_SIZE = 224

check = os.path.exists(path)
if check ==True:
    print('Loading the image from the Folder {}'.format(path))
else:
    print('This path is not exist')
    sys.exit()


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


# A method to count the total number of files
def count_files(path_name):
    ls = os.listdir(path_name)
    count = 0
    for i in ls:
        if os.path.isfile(os.path.join(path_name, i)):
            count += 1
    return count

# Get the path name
name = path.split(os.path.sep)[-1]
# print(name)

def Createfolder(path):
    file_name = ['/real', '/fake']
    for name in file_name:
        os.mkdir(path + name)

def CreateRealDataSet(CATEGORIES,path,StoreDir,IMG_SIZE):
    for category in CATEGORIES:  # Search all category
        IMG_path = os.path.join(path, category)
        for img in os.listdir(IMG_path):
            # Combine the real image path together.
            filepath = os.path.join(IMG_path, img)
            Client_img = cv2.imread(os.path.join(IMG_path, img))
            # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gery', grey)
            # cv2.waitKey()
            print(filepath)
            Macfile = '.DS_Store'
            Thumbsfile = 'Thumbs.db'
            if Macfile in filepath:
                os.remove(filepath)
                continue
            if Thumbsfile in filepath:
                os.remove(filepath)
                continue
            # print(img)
            # normalization the image in the same size.
            Face_image = cv2.resize(Client_img, (IMG_SIZE, IMG_SIZE))
            # Create the save name
            img_name = "%s/%s" % (StoreDir,img)
            cv2.imwrite(img_name, Face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # cv2.imshow('image', image)
            # cv2.waitKey()
            print("Save success!!!")


# Create fake dataset
def CreateFakeDataSet(CATEGORIES,fake_path,Fake_StoreDir,IMG_SIZE):
    for category in CATEGORIES:  # Search all category
        IMG_path = os.path.join(fake_path, category)
        for img in os.listdir(IMG_path):
            # Combine the real image path together.
            filepath = os.path.join(IMG_path, img)
            Client_img = cv2.imread(os.path.join(IMG_path, img))
            # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('gery', grey)
            # cv2.waitKey()
            print(filepath)
            Macfile = '.DS_Store'
            Thumbsfile = 'Thumbs.db'
            if Macfile in filepath:
                os.remove(filepath)
                continue
            if Thumbsfile in filepath:
                os.remove(filepath)
                continue
            # print(img)
            # normalization the image in the same size.
            Face_image = cv2.resize(Client_img, (IMG_SIZE, IMG_SIZE))
            # Create the save name
            img_name = "%s/%s" % (Fake_StoreDir, img)
            cv2.imwrite(img_name, Face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # cv2.imshow('image', image)
            # cv2.waitKey()
            print("Save success!!!")

if __name__ == "__main__":
    # Step1: Clearing the dataset folder..

    # delete_files(Dir)

    # delete_files(Dir2)

    # Step2: Create fake, real folder.
    Createfolder(Dir)
    Createfolder(Dir2)

    CreateRealDataSet(Traing_Real_CATEGORIES,path,StoreDir,IMG_SIZE)

    CreateFakeDataSet(Traing_Fake_CATEGORIES,fake_path,Fake_StoreDir,IMG_SIZE)

    CreateRealDataSet(Test_Real_CATEGORIES, path, Test_StoreDir, IMG_SIZE)

    CreateFakeDataSet(Test_Fake_CATEGORIES, fake_path, Test_Fake_StoreDir, IMG_SIZE)

    realfileNum = count_files(StoreDir)
    fakefileNum = count_files(Fake_StoreDir)

    Test_realfileNum = count_files(Test_StoreDir)
    Test_fakefileNum = count_files(Test_Fake_StoreDir)

    print("There are %d" % (realfileNum) + "real image for Training have been saved")
    print("There are %d" % (fakefileNum) + "fake image for Training have been saved")

    print("There are %d" % (Test_realfileNum) + "real image for Testing have been saved")
    print("There are %d" % (Test_fakefileNum) + "fake image for Testing have been saved")