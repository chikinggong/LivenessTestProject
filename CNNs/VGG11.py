# Import some necessary library
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

'''
This kind of network architecture is reference from paper: 
LiveNet: Improving features generalization for face liveness detection using convolution neural networks.
'''

class LivenessNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # Conv layer 1
        # Which is a 3X3 filer with 64 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=inputShape))
        # Activation ReLU
        model.add(Activation('relu'))
        # Pooling layer 1(max-pooling) 2X2 filter
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Conv layer 2
        # Which is a 3X3 filer with 128 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(128, (3, 3), padding='same'))
        # Activation ReLU
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # Pooling layer 2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Dropout 0.5
        model.add(Dropout(0.5))

        # Conv layer 3.1
        # Which is a 3X3 filer with 256 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Conv layer 3.2
        # Which is a 3X3 filer with 256 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Pooling layer 3
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Conv layer 4.1
        # Which is a 3X3 filer with 128 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Conv layer 4.2
        # Which is a 3X3 filer with 128 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Pooling layer 4
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Conv layer 5.1
        # Which is a 3X3 filer with 128 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Conv layer 5.2
        # Which is a 3X3 filer with 512 kernel.
        # the padding methods is padding by the same boundary's pixel .
        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))

        # Pooling layer 5
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # Fully connected layer 1
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dense(4096))
        model.add(Activation('relu'))

        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model