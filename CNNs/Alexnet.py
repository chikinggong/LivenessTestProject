# author：jiangchiying
# date：2019-07-25 13:23
# tool：PyCharm
# Python version：3.7.1

# Import some necessary library
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

'''

Reference: 

This kind of Convolutional neural networks architecture is reference from paper: 

Krizhevsky, A., Sutskever, I. and Hinton, G. (2012). ImageNet classification with deep convolutional neural networks. Communications of the ACM, 60(6), pp.84-90.

'''

class Alexnet_NetWork:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # Conv layer 1
        # Which is a 11X11 filer with 64 kernel with strides is 4.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(64, (11, 11), strides=(4, 4), padding='valid', input_shape=inputShape))
        # Activation ReLU
        model.add(Activation('relu'))
        # Pooling layer 1(max-pooling) 2X2 filter
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='valid'))


        # Conv layer 2
        # Which is a 11X11 filer with 256 kernel with strides is 1
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
        # Activation ReLU
        model.add(Activation('relu'))

        # Pooling layer 2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # Dropout 0.5
        # model.add(Dropout(0.5))

        # Conv layer 3
        # Which is a 3X3 filer with 384 kernel with strides is 1.
        # the padding methods is padding by the zero boundary's pixel .
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        # Conv layer 4
        # Which is a 3X3 filer with 256 kernel.
        # the padding methods is padding by the zero boundary's pixel .
        model.add(Conv2D(384, (3, 3),strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        # Conv layer 5

        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
        model.add(Activation('relu'))

        # Pooling layer 5(max-pooling) 2X2 filter
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

        # Fully connected layer 1
        model.add(Flatten())
        model.add(Dense(4096, input_shape=(224*224*3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))

        # Fully connected layer 2
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(classes, activation='softmax'))

        return model



