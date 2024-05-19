from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from maping import OutputNeurons
from datagen import training_set, test_set

classifier = Sequential()
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation="relu"))
classifier.add(MaxPool2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(64, activation="relu"))
classifier.add(Dense(OutputNeurons, activation="softmax"))
classifier.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

import time

StartTime = time.time()
classifier.fit_generator(
    training_set,
    steps_per_epoch=30,
    epochs=10,
    validation_data=test_set,
    validation_steps=10,
)
EndTime = time.time()
print("###### Total Time Taken: ", round((EndTime - StartTime) / 60), "Minutes ######")
print("[INFO]: Saving....")
classifier.save("lenet1.hdf5")
classifier.summary()
