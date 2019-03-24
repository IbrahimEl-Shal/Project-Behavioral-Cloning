import csv 
import cv2

lines = []
Flag = True

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
   
images = []
measurements = []

correction = 0.1  

for line in lines:
    for i in range(0,3):
        source_path = line[i]
        token = source_path.split('/')
        filename = token[-1]
        local_path = './data/IMG/' + filename
        image = cv2.imread(local_path)
        images.append(image)
        
    if(Flag):
        measurement = line[3]
        Flag = False
    else:
        measurement = float(line[3])
        measurements.append(measurement)
        measurements.append(measurement-correction)
        measurements.append(measurement-correction)
        
del images[0:3] 
        
augmented_images = []    
augmented_measurements = []

#To deal with pic on opposite direction        
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    filpped_image = cv2.flip(image,1)
    filpped_measurements =  measurement * (-1.0)
    augmented_images.append(filpped_image)
    augmented_measurements.append(filpped_measurements)

import numpy as np

x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
   
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x : (x / 255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((74,24), (0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=3)

model.save('Model.h5')