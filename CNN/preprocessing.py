import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class CNNModel(object):
    def __init__(self):
        self.cnn = None
        self.train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.train_set = self.train_datagen.flow_from_directory(
                './datasets/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')
        self.test_set = self.test_datagen.flow_from_directory(
                './datasets/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

    def model(self):
        self.cnn = Sequential()
        self.cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
        self.cnn.add(MaxPool2D(pool_size=2, strides=1))
        self.cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(MaxPool2D(pool_size=2, strides=1))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(units=128, activation='relu'))
        self.cnn.add(Dense(units=1, activation='sigmoid'))
        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.cnn


