from keras.models import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.core.activation import Activation
from keras.src.layers.core.dense import Dense
from keras.src.layers.reshaping.flatten import Flatten
from keras import backend as K
from maping import OutputNeurons

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Khởi tạo model
        model = Sequential()
        input_shape = (height, width, depth)

        # sử dụng 'channels-first' để input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # Thứ nhất: CONV => RELU => POOL layers
        # model.add(Conv2D(100, (3, 3), padding='same', input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(OutputNeurons,activation="softmax"))

        # Thứ hai: CONV => RELU => POOL layers
        # model.add(Conv2D(100, (3, 3), padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # # Làm phẳng rồi đưa vào lớp FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(500))
        # model.add(Activation('relu'))

        # # Softmax classifier
        # model.add(Dense(classes))
        # model.add(Activation('softmax'))

        # Trả về model (Mạng CNN lenet)
        return model
