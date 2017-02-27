import csv
import os
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import BatchNormalization,Input, Dropout,Activation
import sklearn
from sklearn.utils import shuffle
import random

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

def rotateImage(image):
    rows,cols,channel = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), random.uniform(-3, 3), 1)
    return cv2.warpAffine(image,M,(cols,rows), borderMode=1)

def brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * (1 + np.random.uniform(-0.4, 0.2))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)    

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                correction = 0.05    
                angle = float(batch_sample[3]) 
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    im = rotateImage(image)
                    images.append(im)
                correction = 0.05    
                angle = float(batch_sample[3]) 
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)    
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    brightImage = brightness(image)
                    images.append(brightImage)
                correction = 0.05    
                angle = float(batch_sample[3]) 
                angles.append(angle)
                angles.append(angle + correction)
                angles.append(angle - correction)           

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5,border_mode='valid',activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3, validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')