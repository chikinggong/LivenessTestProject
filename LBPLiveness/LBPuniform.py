# author：jiangchiying
# date：2019-07-16 11:05
# tool：PyCharm
# Python version：3.7.1

# import the necessary packages
import numpy as np
import cv2
from PIL import Image
from pylab import*

'''

Reference :

The ide of extract the LBP uniform feature and calculate the histogram from a single image.

https://github.com/michael92ht/LBP

https://blog.csdn.net/heli200482128/article/details/79204008


'''

class LBPtest:
    def __init__(self):
        # map of uniform LBP
        self.uniform_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 12: 8,
                            14: 9, 15: 10, 16: 11, 24: 12, 28: 13, 30: 14, 31: 15, 32: 16,
                            48: 17, 56: 18, 60: 19, 62: 20, 63: 21, 64: 22, 96: 23, 112: 24,
                            120: 25, 124: 26, 126: 27, 127: 28, 128: 29, 129: 30, 131: 31, 135: 32,
                            143: 33, 159: 34, 191: 35, 192: 36, 193: 37, 195: 38, 199: 39, 207: 40,
                            223: 41, 224: 42, 225: 43, 227: 44, 231: 45, 239: 46, 240: 47, 241: 48,
                            243: 49, 247: 50, 248: 51, 249: 52, 251: 53, 252: 54, 253: 55, 254: 56,
                            255: 57}

    def ImagePre_Proccessing(self, input_image):
        """

        Loading the normalized image and convert it into gray image. Then get the image array

        Args:
            input_image : input the normalized image.

        Returns:

            An array list of the gray image.

        """

        image = cv2.imread(input_image)

        image = cv2.resize(image, (64, 64))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray_image


    def Calute_Basic_LBP(self, image_array, i, j):
        '''

        :param image_array: the Input image array which need to be calculate the lbp.
        :param i: The Point in the x direction of the coordinate axis.
        :param j: The Point in the Y direction of the coordinate axis
        :return: the sum list of a
        '''
        sum = []

        # top-left
        if image_array[i-1,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)

        # mid-left
        if image_array[i-1,j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)

        # left-bottom
        if image_array[i-1,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        #  mid-top
        if image_array[i,j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        #  mid-bottom
        if image_array[i,j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        # rigth-top
        if image_array[i+1, j-1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        #     mid-right
        if image_array[i+1, j]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)
        # right-bottom
        if image_array[i+1, j+1]>image_array[i,j]:
            sum.append(1)
        else:
            sum.append(0)

        lbp_sum = sum[0] * 1 + sum[1] * 2 + sum[2] * 4 + sum[3] * 8 + sum[4] * 16 + sum[5] * 32 + sum[6] * 64 + sum[7] * 128

        return lbp_sum


    def calc_sum(self, r):
        num = 0
        while (r):
            r &= (r - 1)
            num += 1
        return num

    def LBP_Basic(self, gray_image):

        """

               Get the Basic LBP feature from the image.

               Args:
                   image_array : the array of the gray image.

               Returns:

                   An array list of the Basic LBP feature.

         """
        # Create a empty array list prepare to store the feature
        lbp_basic_array = np.zeros(gray_image.shape, np.uint8)

        # Get the image width and height from the input image.
        np_width = gray_image.shape[0]
        np_height = gray_image.shape[1]

        # Start to scan the Gray image to calcualte the lbp values.

        for i in range(1, np_width - 1):
            for j in range(1, np_height - 1):
                lbp_sum = self.Calute_Basic_LBP(gray_image, i, j)

                lbp_basic_array[i, j] = lbp_sum

        # print(lbp_basic_array)
        return lbp_basic_array


    # Get the LBP uniform from the gray image
    def lbp_uniform(self,gray_image):
        uniform_array=np.zeros(gray_image.shape, np.uint8)
        basic_array=self.LBP_Basic(gray_image)

        # Get the image width and height from the input image.
        width=gray_image.shape[0]
        height=gray_image.shape[1]

        for i in range(1,width-1):
            for j in range(1,height-1):
                 k= basic_array[i,j]<<1
                 if k>255:
                     k=k-255
                 xor=basic_array[i,j]^k
                 num=self.calc_sum(xor)
                 if num<=2:
                     uniform_array[i,j]=self.uniform_map[basic_array[i,j]]
                 else:
                     uniform_array[i,j]=58
        return uniform_array


    # Calculate the histogram
    def Calculate_hist(self,img_array):
        (hist, _) = np.histogram(img_array.ravel(),
                                 bins=np.arange(0, 60),
                                 range=(0, 60))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum()+1e-7)

        return hist

    # Show the image.
    def show_image(self, image_array):
        cv2.imshow('Image', image_array)
        cv2.waitKey(0)

    # Show the histogram
    def show_hist(self, img_array, im_bins, im_range):
        hist = cv2.calcHist([img_array], [0], None, im_bins, im_range)
        hist = cv2.normalize(hist,hist).flatten()
        plt.plot(hist, color='r')
        plt.xlim(im_range)
        plt.show()


if __name__ == '__main__':
    imagepath = "../NUAAdataset/fake/0001_00_00_01_0.jpg"
    lbp = LBPtest()
    image_array = lbp.ImagePre_Proccessing(imagepath)
    basic_array=lbp.LBP_Basic(image_array)
    # lbp.show_image(basic_array)
    lbp.show_image(basic_array)
    uniform_array=lbp.lbp_uniform(image_array)
    # print(uniform_array)
    hist = lbp.Calculate_hist(uniform_array)
    # lbp.show_hist(uniform_array,[60],[0,60])
    print(hist)
