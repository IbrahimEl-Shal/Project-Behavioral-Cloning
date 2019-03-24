# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we used what we've learned about deep neural networks and convolutional neural networks to clone driving behavior by train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

With using a simulator we you can steer a car around a track for data collection. I used image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:
- [Keras](https://keras.io/)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](http://matplotlib.org/) 

### Data 
I used the data which was provided with the project. The simulator at a given time step  recordes three images taken from left, center, and right cameras. The following figure shows an examples : 

Left| Center | Right
----|--------|-------
![left](./Images/left.jpg) | ![center](./Images/center.jpg) | ![right](./Images/right.jpg)

### Model Architecture and Training Strategy

#### 1. Normalization:
In Keras, lambda layers can be used to create arbitrary functions that operate on each image as it passes through the layer. So i used it to normalize the data.

#### 2. Cropping Images:
The top portion of the image captures trees and hills and sky and the bottom portion of the image captures the hood of the car. We have to crop the useless parts to get the useful informations.
Keras provides the Cropping2D layer for image cropping.

#### 3. Convolution Neural Network and Activation Function : 
I built a convolution neural network with 5x5 filter sizes and RELU layers as a activation function to introduce nonlinearity. 

#### 4. Attempts to reduce overfitting in the model
The model was trained and validated on different data sets to ensure that the model was not overfitting and I used the early stoping method to avoid overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 5. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually.

#### 6. Solution Design Approach
Architecture is similar to the LeNet.
I split the images and steering angle data into a training and validation data set.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

####  My final model architecture : 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   			    		| 
| Lambda     	        | Normalization                                 |
| Cropping2D          	| 74 from top and 24 from bottom                |
| Convolution 5x5     	| 2x2 stride with 24 output filters             |
| RELU					| Activation Function							|
| Convolution 5x5     	| 2x2 stride with 36 output filters          	|
| RELU					| Activation Function							|
| Convolution 5x5     	| 2x2 stride with 48 output filters             |
| RELU					| Activation Function							|
| Convolution 3x3     	| 1x1 stride with 64 output filters            	|
| RELU					| Activation Function							|
| Convolution 3x3     	| 1x1 stride with 64 output filters            	|
| RELU					| Activation Function							|
| Flatten				|		            							|
| Dropout				| 0.5		            						|
| Fully connected		| output 100						 			|
| RELU     			    | Activation Function				  	    	|
| Fully connected		| output 50 		      		     			|
| RELU     				| Activation Function			  		    	|
| Fully connected		| output 10 		      		     			|
| RELU     				| Activation Function			  		    	|
| Fully connected		| output 1 		 	     		     			|

####  Architecture Summary: 

Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 62, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_2 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0