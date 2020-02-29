import os
import tensorflow as tf
import cv2
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
folder='/home/manik/Documents/images/'
photos, labels=list(), list()
for file in listdir(folder):
        print(file)
        output=0.0
        if file.startswith('Manik'):
                output=1.0
        #photo=cv2.imread(folder+ file)
        photo=load_img(folder+ file,target_size=(150,150))
        #cv2.imshow('img',photo)
        #print(type(photo))
        #photo=img_to_array(photo)
        #photo=np.asarray(photo)
        photo=np.resize(photo,(-1,(150,150)))
        #print(photo)
        photos.append(photo)
        labels.append(output) 
#photos=photos.reshape(3,150,150,3)     
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

history = model.fit(
      x=photos,y=labels,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1)
photo_test=load_img('/home/manik/Documents/Current_images',target_size=(150,150))      
print(model.predict(photo_test))     
