import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model

samples = []
with open('./data_mouse/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

##Notice how I am grabbing the left/right camera images as well. The value camera_angle_offset determines how much steering
##angle should be applied given the offset. I had a lack of data with large steering angles, so I flipped all images/angles with
##a steering angle greater than 0.2 and added them to the training data set.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Used as a reference pointer so code always loops back around
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            camera_angle_offset = 0.1
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data_mouse/IMG/' + batch_sample[0].split('\\')[-1]
                left_name = './data_mouse/IMG/' + batch_sample[1].split('\\')[-1]
                right_name = './data_mouse/IMG/' + batch_sample[2].split('\\')[-1]
                center_image = cv2.imread(center_name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.imread(left_name)
                left_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.imread(right_name)
                right_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + camera_angle_offset
                right_angle = float(batch_sample[3]) - camera_angle_offset
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                if abs(center_angle) > 0.2:
                    center_image_flipped = np.fliplr(center_image)
                    center_reversed_angle = -center_angle
                    left_image_flipped = np.fliplr(left_image)
                    left_reversed_angle = -left_angle
                    right_image_flipped = np.fliplr(right_image)
                    right_reversed_angle = -right_angle
                    images.append(center_image_flipped)
                    angles.append(center_reversed_angle)
                    images.append(left_image_flipped)
                    angles.append(left_reversed_angle)
                    images.append(right_image_flipped)
                    angles.append(right_reversed_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            X_train = X_train[:,65:145,:,:] 
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
			
additionalDataCount = 0
for line in train_samples:
    if abs(float(line[3])) > 0.2:
        additionalDataCount+=1
additionalDataCount *= 3 #Multiply by 3 because I use left/right camera images as well.

total_samples = len(train_samples)*3 + additionalDataCount
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch, = 80, 320, 3  # Trimmed image format

#Tried to replicate the network described in the following article: http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
#I don't like reducing the depth of fully connected layers too quickly, so I opted to make the step down more gradual

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu', name = 'Conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu', name = 'Conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu', name = 'Conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name = 'Conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name = 'Conv5'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
model.add(Flatten())
model.add(Dense(500, activation='relu', name='FC1'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu', name='FC2'))
model.add(Dropout(0.4))
model.add(Dense(10, name='FC3'))
model.add(Dense(1))
model.summary()


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=total_samples, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=6, initial_epoch=0)
model.save('./model.h5')


