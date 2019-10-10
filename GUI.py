# author：jiangchiying
# date：2019-07-18 15:17
# tool：PyCharm
# Python version：3.7.1

# import the necessary library
import tkinter
from tkinter import filedialog, messagebox, scrolledtext, Frame
import os
import cv2
import PIL.Image, PIL.ImageTk
from Detector import Detector
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import logging
import pickle
from LBPLiveness.LBPuniform import *
import time


logging.getLogger('tensorflow').disabled = True
WindowWidth = 2000
WindowHeight = 1000
image_label=['fake', 'real']
Image_size = 224

class Application():
    '''

    Liveness tests for face recognition Application GUI.

    '''
    def __init__(self, master=None):
        # Initialize GUI window
        self.root = tkinter.Tk()


        # Set the title name of the window.
        self.root.title('Liveness Tests System')

        # Set the size of the window.
        self.root.geometry(
            "%dx%d" % (WindowWidth, WindowHeight)
        )
        # Create title
        self.Create_title()

        # Add button
        self.button()

        # set menu
        menubar = tkinter.Menu(self.root)
        self.root.config(menu=menubar)

        # Create About menu bar.
        about_me = tkinter.Menu(menubar)
        about_me.add_command(label='About me', command=self.__about_me__)
        menubar.add_cascade(label='About', menu=about_me)

    # Set the title in the window
    def Create_title(self):
        fm1 = tkinter.Frame(self.root, bg='black')
        Label_title = tkinter.Label(fm1, text='Liveness Tests For Face recognition', font=('微软雅黑', 24), fg="white",
                                    bg='black')
        Label_title.pack()
        fm1.pack(side="top", padx=10, pady=10)

    #  Button
    def button(self):

        # Create
        input_image_frame = tkinter.Frame(self.root)
        input_image_lb = tkinter.Label(input_image_frame, text="Please upload a human face photo for liveness test", font=('微软雅黑', 20), fg="black")
        input_image_lb.pack()
        input_image_frame.pack(side="top", padx=10, pady=10)
        file_button = tkinter.Button(
            input_image_frame, text='Upload Photo',height=2, width=15,
            command=self.__get_file__)
        file_button.pack(side=tkinter.LEFT)
        # Create upload video button
        video_button = tkinter.Button(
            input_image_frame, text='Upload Video', height=2, width=15,
            command=self.__get_video__)
        video_button.pack(side=tkinter.LEFT)

        # Create Clear data button.
        clear_data_button = tkinter.Button(
            input_image_frame, text='Clear Data', height=2, width=15, command=self.__clear_data__)
        clear_data_button.pack(side=tkinter.LEFT)

        # Create Face detect button.
        face_detect_button = tkinter.Button(
            input_image_frame, text='Face detection', height=2, width=15, command=self.__face_detect__)
        face_detect_button.pack(side=tkinter.LEFT)

        # Create CNN Detect button.
        CNN_button = tkinter.Button(
            input_image_frame, text='CNN Liveness test', height=2, width=15, command=self.__CNN_test__)
        CNN_button.pack(side=tkinter.LEFT)

        # Create LBP Detect button.
        LBP_button = tkinter.Button(
            input_image_frame, text='LBP Liveness test', height=2, width=15, command=self.__LBP_test__)
        LBP_button.pack(side=tkinter.LEFT)

        # Create Real time Detect button.
        Real_time_button = tkinter.Button(
            input_image_frame, text='Real time Liveness test demo', height=2, width=25, command=self.__Real_time__)
        Real_time_button.pack(side=tkinter.LEFT)



    # Get the upload
    def __get_file__(self):
        file_dir = filedialog.askopenfilename()
        filetype = os.path.splitext(file_dir)[1]

        if len(file_dir) > 0:
            if filetype in ['.jpg', '.png', '.jpeg', '.bmp']:
                # messagebox.showerror('error', message=filetype)
                self.__show_img__(file_dir)

            else:
                messagebox.showerror('error', message='The file type is wrong, please upload .jpg, .png, .jpeg, .bmp file')

    # Display the image
    def __show_img__(self, photo_dir):
        global panelA
        # Read the image by Opencv.

        self.image = cv2.imread(photo_dir)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # Get the image dimensions
        # height, width, no_channels = image.shape
        image = PIL.Image.fromarray(image)
        image = PIL.ImageTk.PhotoImage(image)
        # if the panels are None, initialize them
        if panelA is None:
            # the first panel will store our original image
            Upload_image_frame = tkinter.Frame(self.root)
            Upload_image_lb = tkinter.Label(Upload_image_frame, text="This is the uploaded image.", font=('微软雅黑', 20), fg="black")
            Upload_image_lb.pack()
            Upload_image_frame.pack(side="left", fill='y', padx=10, pady=10)
            panelA = tkinter.Label(Upload_image_frame, image=image)
            panelA.image = image
            panelA.pack()



        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image


    def __get_video__(self):
        file_dir = filedialog.askopenfilename()
        filetype = os.path.splitext(file_dir)[1]

        if len(file_dir) > 0:
            if filetype in ['.mov']:
                model = load_model("ModelandPickle/AlexnetRP.model")
                # Open the computer web cam.
                cap = cv2.VideoCapture(file_dir)

                while True:
                    # Capture the frame from the web camera.
                    ret, frame = cap.read()

                    facedector = Detector()
                    face = facedector.detect_face(frame)
                    for (x, y, w, h) in face:

                        color = (0, 0, 255)
                        stroke = 2
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
                        # Capture the  face from the frame and then test it
                        Capture_face = frame[y: y + h, x + 10: x + w - 10]
                        Capture_face = cv2.resize(Capture_face, (Image_size, Image_size))
                        Capture_face = Capture_face.astype("float")
                        Capture_face = img_to_array(Capture_face)
                        face = np.expand_dims(Capture_face, axis=0)
                        preds = model.predict(face)[0]
                        j = np.argmax(preds)
                        result = int(j)
                        label = image_label[result]
                        if (label == 'real'):
                            label = "{}".format(label)
                            cv2.putText(frame, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h),
                                          (0, 255, 0), 2)
                        elif (label == 'fake'):
                            label = "{}".format(label)
                            cv2.putText(frame, label, (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.rectangle(frame, (x, y), (x + w, y + h),
                                          (0, 0, 255), 2)

                    frame = cv2.putText(frame, 'Press Q to exit', (0, 10), cv2.FONT_HERSHEY_COMPLEX, 1.2,
                                        (255, 255, 255), 1)
                    # Display the real time frame.
                    cv2.imshow('Face Detection', frame)

                    # If user type Q from the keyboard the loop could be break.
                    if cv2.waitKey(20) & 0xFF == ord('q'):
                        break

                # Release the resource and destroy all windows.
                cap.release()
                cv2.destroyAllWindows()

            else:
                messagebox.showerror('error',
                                     message='The file type is wrong, please upload .mov file')
    # Face detect method
    def __face_detect__(self):
        global panelA
        global panelB
        if panelA is None:
            messagebox.showerror('error', message='No image has been upload..')
        if self.image is None:
            messagebox.showerror('error', message='No image has been upload..')
        else:
            facedector = Detector()
            # print(self.image)
            face = facedector.detect_face(self.image)
            if len(face) > 0:
                for (x, y, w, h) in face:
                    color = (0, 0, 255)
                    stroke = 2
                    # cv2.rectangle(self.image, (x, y), (x + w, y + h), color, stroke)
                    image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    # Get the image dimensions
                    height, width, no_channels = image.shape
                    image = PIL.Image.fromarray(image)
                    image = PIL.ImageTk.PhotoImage(image)
                    panelA.configure(image=image)
                    panelA.image = image

                    self.Crop_image = self.image[y: y + h, x: x + w]
                    Crop_image = cv2.cvtColor(self.Crop_image, cv2.COLOR_BGR2RGB)
                    Crop_image = PIL.Image.fromarray(Crop_image)
                    Crop_image = PIL.ImageTk.PhotoImage(Crop_image)

                    if panelB is None:
                        face_detect_frame = tkinter.Frame(self.root)
                        face_image_lb = tkinter.Label(face_detect_frame, text="The Crop Image",
                                                        font=('微软雅黑', 20), fg="black")
                        face_image_lb.pack()
                        panelB = tkinter.Label(face_detect_frame, image=Crop_image)
                        panelB.image = Crop_image
                        panelB.pack()
                        # face_detect_frame.pack(side="left", fill='y',padx=10, pady=10)
                        face_detect_frame.pack(anchor='nw',padx=10, pady=10)

                    # otherwise, update the image panels
                    else:
                        # update the pannels
                        panelB.configure(image=Crop_image)
                        panelB.image = Crop_image

            else:
                messagebox.showerror('error', message='No face has been detected..')


    # CNN test method.
    def __CNN_test__(self):
        global panelB
        global CNN_result_lb
        if panelB is None:
            messagebox.showerror('error', message='No Crop image, Liveness detect fail...')
        if self.Crop_image is None:
            messagebox.showerror('error', message='No Crop image, Liveness detect fail...')
        else:
            # load the liveness detector model and label encoder from disk

            print("[INFO] loading liveness det"
                  "ector...")
            model = load_model("ModelandPickle/Alexnetv2.model")
            time_start = time.time()
            test_image = self.Crop_image
            test_image = cv2.resize(test_image, (Image_size, Image_size))
            test_image = test_image.astype("float")
            test_image = img_to_array(test_image)
            face = np.expand_dims(test_image, axis=0)
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            # label = le.classes_[j]
            result = int(j)
            # print(result)
            label = image_label[result]
            label = "{}: {:.3f}".format(label, preds[j])

            # messagebox.showinfo('Result', message="CNN Liveness test result is :  "+label)
            # print(label)
            if CNN_result_lb is None:
                Liveness_result_frame = tkinter.Frame(self.root)
                CNN_result_lb = tkinter.Label(Liveness_result_frame, text="CNN Liveness test result is :  "+label,
                                              font=('微软雅黑', 20), fg="red")
                CNN_result_lb.pack()
                Liveness_result_frame.pack(side="bottom", fill='x',padx=10, pady=10)
                time_end = time.time()
                time_cost = time_end-time_start
                print("CNN Time cost is ",time_cost)
            else:
                # update the pannels
                CNN_result_lb.configure(text="CNN Liveness test result is :  "+label)
                CNN_result_lb.text = "CNN Liveness test result is :    "+label

    def __clear_data__(self):
        global panelB, panelA, CNN_result_lb, LBP_result_lb, panelC

        # if the user has uploaded the photo.
        if not panelA is None:
            panelA.configure(image=None)
            panelA.image = None
            self.image = None
            self.Crop_image = None



        if not panelB is None:

            panelB.configure(image=None)
            panelB.image = None

        if not panelC is None:

            panelC.configure(image=None)
            panelC.image = None

        if not CNN_result_lb is None:
            CNN_result_lb.configure(text="")
            CNN_result_lb.text = ""


        if not LBP_result_lb is None:
            LBP_result_lb.configure(text="")
            LBP_result_lb.text = ""

    def __LBP_test__(self):
        global panelB
        global LBP_result_lb
        global panelC
        if panelB is None:
            messagebox.showerror('error', message='No Crop image, Liveness detect fail...')
        if self.Crop_image is None:
            messagebox.showerror('error', message='No Crop image, Liveness detect fail...')
        else:
            # load the liveness detector model
            time_start=time.time()
            lbp = LBPtest()
            Testingdata = []

            crop_image = self.Crop_image
            gray_image = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)
            basic_array = lbp.LBP_Basic(gray_image)
            LBP_result = PIL.Image.fromarray(basic_array)
            LBP_img = PIL.ImageTk.PhotoImage(LBP_result)

            if panelC is None:
                LBP_result_frame = tkinter.Frame(self.root)
                LBP_image_lb = tkinter.Label(LBP_result_frame, text="Image LBP feature", font=('微软雅黑', 15), fg="black")
                LBP_image_lb.pack()
                panelC = tkinter.Label(LBP_image_lb, image=LBP_img)
                panelC.image = LBP_img
                panelC.pack()
                LBP_result_frame.pack(side="left", fill='x',padx=10, pady=10)
                time_end = time.time()
                time_cost = time_end-time_start
                print("the LBP time cost is ", time_cost)

            # otherwise, update the image panels
            else:
                # update the pannels
                panelC.configure(image=LBP_img)
                panelC.image = LBP_img


            file = open('ModelandPickle/LBPSVM3.pickle','rb')
            s = file.read()
            model = pickle.loads(s)
            test_image = self.Crop_image
            test_image = cv2.resize(test_image, (64, 64))
            image_array = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            uniform_array = lbp.lbp_uniform(image_array)
            hist = lbp.Calculate_hist(uniform_array)
            Testingdata.append(hist)
            prediction = model.predict(Testingdata)
            # Show the percentage of prediction
            Real_percentage = model.predict_proba(Testingdata)[0][1]
            percentage = model.predict_proba(Testingdata)[0]
            Fake_percentage = model.predict_proba(Testingdata)[0][0]
            result = prediction[0]
            print(percentage)
            if(result=='fake'):
                result = "{}: {:.3f}".format(result, Fake_percentage)

            if (result == 'real'):
                result = "{}: {:.3f}".format(result, Real_percentage)

            # print(label)
            if LBP_result_lb is None:
                Liveness_result_frame = tkinter.Frame(self.root)
                LBP_result_lb = tkinter.Label(Liveness_result_frame, text="LBP test result is :    " + result,
                                              font=('微软雅黑', 20), fg="red")
                LBP_result_lb.pack()
                Liveness_result_frame.pack(side="bottom", fill='x', padx=10, pady=10)
            else:
                # update the pannels
                LBP_result_lb.configure(text="LBP test result :    " + result)
                LBP_result_lb.text = "Result is :    " + result
            # messagebox.showinfo('Result', message="LBP Liveness test result is :  " + result)
    def __Real_time__(self):
        # Loading the Realtime liveness test model...
        model = load_model("ModelandPickle/Realtimev2.model")
        # le = pickle.loads(open('Userpickle2.pickle', "rb").read())

        # Open the computer web cam.
        cap = cv2.VideoCapture(0)

        while True:
            # Capture the frame from the web camera.
            ret, frame = cap.read()

            facedector = Detector()
            face = facedector.detect_face(frame)
            for (x, y, w, h) in face:

                color = (0, 0, 255)
                stroke = 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)
                # Capture the  face from the frame and then test it
                Capture_face = frame[y + 10: y + h - 10, x + 10: x + w - 10]
                Capture_face = cv2.resize(Capture_face, (Image_size, Image_size))
                Capture_face = Capture_face.astype("float")
                Capture_face = img_to_array(Capture_face)
                face = np.expand_dims(Capture_face, axis=0)
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                result = int(j)
                label = image_label[result]
                if (label == 'real'):
                    label = "{}: {:.3f}".format(label, preds[j])
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                elif (label == 'fake'):
                    label = "{}: {:.3f}".format(label, preds[j])
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255), 2)

            frame = cv2.putText(frame, 'Press Q to exit', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
            # Display the real time frame.
            cv2.imshow('Face Detection', frame)

            # If user type Q from the keyboard the loop could be break.
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        # Release the resource and destroy all windows.
        cap.release()
        cv2.destroyAllWindows()

    # The about me command method.
    def __about_me__(self):
        messagebox.showinfo(
            title='About Me',
            message='Author: Chiying Jiang\r\n'
                    'GitHub: https://github.com/chikinggong')

    # The method to start the application.
    def start_Application(self):
        self.root.mainloop()



if __name__ == '__main__':

    system = Application()
    panelA = None
    panelB = None
    panelC = None
    CNN_result_lb = None
    LBP_result_lb = None

    system.start_Application()