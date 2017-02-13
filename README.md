#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
I did all of my work in Model.ipynb, then transferred the relevant code to model.py for the sake of having an easily runnable python file. The file path to the data is specific to my set up. 

###Model Architecture and Training Strategy

Tried to replicate the network described in the following article (recommended in the lectures): http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
I don't like reducing the depth of fully connected layers too quickly, so I opted to make the step down more gradual.

The model includes RELU layers to introduce nonlinearity (code line 94, 96, 98, 100, 102), and the data is normalized in the model using a Keras lambda layer (code line 90). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 106, 108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 20 for the split, lines 80,81 for processing). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the example set in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf, which was recommended in the lectures

I started by adding the convolution layers with relu activation and max pooling to "stretch" the data space. Then I flattened and added some fully connected layers with dropout to prevent overfitting. Two of the three
fully connected layers also had relu activations. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model performed quite impressively out of the box. Both the training set MSE and validation
set MSE were quite low, which led me to believe there was no over/underfitting going on. One thing I realized I was doing wrong only after hours of head scratching was that I was saving the model before I trained it. Then
I would use my model.h5 file to try to run the simulator, and the car would just list left or right and make no changes. Pretty obvious it wasn't going to work considering it was just applying purely random untrained math to 
generate steering angles. 

The final step was to run the simulator to see how well the car was driving around track one. The first few runs had the car falling off the track during the sharp turns, as well as having trouble driving in the 
center of the road. I realized I needed more training data on sharp turns, as the model wasn't handling them properly. I also realized I should be leveraging the left/right camera images as well to help the car
stay centered in the lane. I fired up the simulator in training mode and gathered more data around tight curves. Also generated more recovery data moving from the outisde of the track to the middle.

I refactored my generator to make use of the left/right camera images, but that introduced a new parameter to tune: steering angle offset. After a few runs, I found what I thought was an appropriate offset, as the car
was finally starting to make it around the track.

####2. Final Model Architecture

The final model architecture (model.py lines 90-111) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 80, 320, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
Conv1 (Convolution2D)            (None, 80, 320, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 40, 160, 24)   0           Conv1[0][0]                      
____________________________________________________________________________________________________
Conv2 (Convolution2D)            (None, 40, 160, 36)   21636       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 20, 80, 36)    0           Conv2[0][0]                      
____________________________________________________________________________________________________
Conv3 (Convolution2D)            (None, 20, 80, 48)    43248       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 10, 40, 48)    0           Conv3[0][0]                      
____________________________________________________________________________________________________
Conv4 (Convolution2D)            (None, 10, 40, 64)    27712       maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 5, 20, 64)     0           Conv4[0][0]                      
____________________________________________________________________________________________________
Conv5 (Convolution2D)            (None, 5, 20, 64)     36928       maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 3, 10, 64)     0           Conv5[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1920)          0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
FC1 (Dense)                      (None, 500)           960500      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 500)           0           FC1[0][0]                        
____________________________________________________________________________________________________
FC2 (Dense)                      (None, 100)           50100       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           FC2[0][0]                        
____________________________________________________________________________________________________
FC3 (Dense)                      (None, 10)            1010        dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             11          FC3[0][0]                        
====================================================================================================
Total params: 1,142,969
Trainable params: 1,142,969
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center_example.jpg]

I then recorded two laps driving in the opposite direction, as the course has much more left turns than right, and I wanted more right-hand turn data.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if it should start to veer off center.

I only grabbed a few data points from track two, as the track was radically different and I wanted to get back to testing my model. If I wanted to get my model working on the second track,
I would definitely need more data points from track two as the turns are MUCH sharper.

I knew I was going to have to augment the data set due to the disproportionately high number of images with 0 steering angle. I decided to flip the image and steering angle for all data points where the steering angle was
greater than 0.2. I didn't want any additional data with low steering angles, as I already had plenty. I generated some histograms visible in Model.ipynb that demonstrate the distribution of data samples, both before and after
generating the additional images.

![flipped_image.jpg]

I also cropped the images to only include the portion relevant to training. This meant eliminating the majority of the front of the car as well as the horizon. I did some testing as to what the perfect values were, and I 
settled with 65:145 slice of the original 180 pixels.

The same changes had to be made to drive.py, as the model was expecting this cropped image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I settled on 6 epochs, even though I think I could have seen a very slight improvement when using 10. At this point,
I was just trying to train faster as I had already spent a lot of time training networks! I used an adam optimizer so that manually training the learning rate wasn't necessary.
