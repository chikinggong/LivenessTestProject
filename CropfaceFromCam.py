# author：jiangchiying
# date：2019-07-02 14:29
# tool：PyCharm
# Python version：3.7.1

# Import some necessary library
import cv2
import os
import shutil
import argparse
import sys

'''

The script for crop face from the web cam.

The command of run this script is: python CropFaceFromCam.py --output Userdataset/(real or fake)

'''
parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument("-o", "--output", type=str, required=True,
                        help="path to output directory of cropped faces")
args = parser.parse_args()
path = args.output
print(path)

check = os.path.exists(path)
if check ==True:
    print('Preparing crop image to {}'.format(path))
else:
    print('This path is not exist')
    sys.exit()

# A method to clear the folder
def delete_files(path_name):
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
    import os
    ls = os.listdir(path_name)
    count = 0
    for i in ls:
        if os.path.isfile(os.path.join(path_name, i)):
            count += 1
    return count

# Call the web camera and take a screenshot of the face
# window_name:  the name of the window.
# camera_index: the index number of camera, 0 is the computer camera, 1 is the extra camera.
# pic_num: the number of the picture need to capture.
# path_name: the save path of the screenshot.

def CapPicFromCam(window_name, camera_index, pic_num, path_name):
    # Set the name of window.
    cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_index)

    # Loading the haarcascade from the FaceDetector folder.
    OSSeparator = os.path.sep
    detectorDir = 'FaceDetector' + OSSeparator + 'haarcascade_frontalface_default.xml'

    # Loading the face detector
    faceCascade = cv2.CascadeClassifier(detectorDir)

    # The color and stroke of the face detector boundary
    color = (0, 0, 255)
    stroke = 2
    # The total number of capture image.
    totalnum = count_files(path)
    pic_num = pic_num + totalnum
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the color image to gray image.
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        # minSize – Minimum possible object size. Objects smaller than that are ignored
        faces = faceCascade.detectMultiScale(grey, scaleFactor = 1.5, minNeighbors = 5, minSize = (32, 32))
        if len(faces) > 0:
            for face in faces:
                x, y, w, h = face

                # Save the face image into Data set file path.
                img_name = "%s/%d.jpg" % (path_name, totalnum)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                totalnum += 1
                # If exccess the max number of capture break out of the loop
                if totalnum > (pic_num):
                    break

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, stroke)

                # Display how many picture have been capture.
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d/100' % (totalnum), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

                # If exccess the max number of capture end the method.
        if totalnum > (pic_num):
            print('Saving success!!')
            break

        frame = cv2.putText(frame, 'Press Q to exit', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # Release the resource and destroy all windows.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # delete_files(path)
    # count_files(path)
    CapPicFromCam("Capture face image", 0, 100, path)
