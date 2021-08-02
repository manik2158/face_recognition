import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import load_img
import numpy as np
import cv2
class smallmodel:
    def build(img_width,img_heights):
# dimensions of our images.
        img_width, img_height = 150, 150

        train_data_dir = '/home/manik/Documents/'
        epochs = 6
        batch_size = 10

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)
        model = Sequential()
        model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
        model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4))
        model.add(Activation('softmax'))



        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')

        model.fit_generator(
            train_generator,
            steps_per_epoch=15 // batch_size,
            epochs=epochs)
        img=cv2.imread("/home/manik/Documents/Current_images/my_image.png")
        img = cv2.resize(img,(150,150))
        img = np.reshape(img,[1,150,150,3])      
        prediction=model.predict_proba(img)
        print(prediction)
        return prediction
