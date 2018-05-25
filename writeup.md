# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/XiangKe1015/German-Traffic-Sign-Classification/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32*32*3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Below are exploratory visualization of the data set. First fig is a picture collection for all classes, and the second picture is a bar chart showing the train image number of each class.
![avatar](/SavedImage/TrainSetVis.png)
![avatar](/SavedImage/DataDistribution.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale is convenient to calculate the gradient and easy for processing.

Here is an example of a traffic sign image before and after grayscaling.
![avatar](/SavedImage/SignRGB.png)   ![avatar](/SavedImage/SignGray.png)

As a last step, I normalized the image data so that the data has mean zero and equal variance. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten			    | output 400									|
| Fully connected layer1| output 120									|
| RELU					| 												|
| Dropout				| 												|
| Fully connected layer2| output 84										|
| RELU					| 												|
| Fully connected layer3| output 43										|
| Softmax				|												|
| Cross Entropy			|			|									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer,with parameters epoch=50, batch-size=64 and learing rate=0.001,

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.948.
* test set accuracy of 0.939.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First, I tried the LeNet architecture, because it's already been proved quite useful for image classification, especially for letter classification.

* What were some problems with the initial architecture?
The accuracy can't meet requirments(0.93), while epoch increase to some degree, the accuracy not increase, seems overfitting occurs.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Here ajust the architeture by adding dropout layer in each converlution layer and first fully connected layer, so the data can be trained with larger epoch, but without overfitting.

* Which parameters were tuned? How were they adjusted and why?
1.Epoch was tuned, to see how the accuracy change with diffrent epoch, this is critical for underfiting and overfiting, and finnaly chosed 50 for epoch.
2.keep_prob was tuned, but seems 0.5 is the best one. 
3.batch-size was tuned, seems 64 is better than 128,due to some class's training sample is relatively small?

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
1.Chose LeNet as start model for this problem, as it's proven successfu in arabic numerals classification, and traffic sign also contains arabic numerals and simple geometric figure.
2.When the LeNet accuracy can't meet requirments, analyze and evaluate the train set distribution, for some class the training set is relatively small, and one way is to increase epoch, but overfitting could also happen, so add dropout in coverlution layer and fully connected layer to prevent overfitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:
![avatar](/NewImage/3.png) ![avatar](/NewImage/11.png) ![avatar](/NewImage/12.png) 
![avatar](/NewImage/13.png) ![avatar](/NewImage/14.png) ![avatar](/NewImage/17.png) 
![avatar](/NewImage/18.png) ![avatar](/NewImage/25.png) ![avatar](/NewImage/38.png) 
![avatar](/NewImage/36.png) ![avatar](/NewImage/34.png) ![avatar](/NewImage/37.png) 

The first image might be difficult to classify because this image is quite close to oher Speed limit sign, the misjudge could happen.The last image might also be difficult to classify, because this class has less train set than other type, which has a a big influence on training accuracy.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image						 |  Prediction	        				 | 
|:--------------------------:|:-------------------------------------:| 
| 3:  60km/h				 | 80km/h   							 | 
| 11：Right-of-way			| Right-of-way 							|
| 12：Priority road			| Priority road							|
| 13：Yield					| Yield					 				|
| 14：Stop 					| Traffic signals						|
| 17：Yield					| Yield									|
| 18：General caution		| General caution		 				|
| 25：Road work				| Road work      						|
| 34：Turn left ahead		| Turn left ahead						|
| 36：Straight or right		| Straight or right						|
| 37：Straight or left		| Go straight or left 					|
| 38：Keep right				| Keep right							|


The model was able to correctly guess 10 of the 12 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 0.939.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

For the first image(class=3), the model is misjudged that this is Speed limit 60km/h sign (probability of 0.46), and the image is Speed limit 60km/h sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       						| 
|:---------------------:|:---------------------------------------------:| 
| .4596 				| Speed limit(80km/h)   						| 
| .2410    				| Speed limit(60km/h)							|
| .1997					| End of speed limit (80km/h)					|
| .0201	      			| Speed limit (30km/h)							|
| .0192				    | Speed limit (50km/h)      					|

For the second image(class=11), the model is quite sure that this is a Right-of-way at the next intersection sign (probability of 0.993), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9933         		| Right-of-way at the next intersection   		| 
| .0065     			| Beware of ice/snow 							|
| .0001					| Priority road, probability					|
| .0001	      			| Traffic signals, probability			 		|
| .0000				    | Dangerous curve to the right, probability		|

For the third image(class=12), the model is quite sure that this is a Priority road sign (probability of 0.9976), and the image does a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9976         		| Priority road   								| 
| .0019     			| Keep right 									|
| .0002					| Ahead only									|
| .0001	      			| Roundabout mandatory			 				|
| .0001				    | End of all speed and passing limits			|

For the fourth image(class=13), the model is quite sure that this is a Yield sign (probability of 1.000), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0000         		| Yield   										| 
| .0000     			| Ahead only 									|
| .0000					| No vehicles									|
| .0000	      			| No passing			 						|
| .0000				    | Keep right									|


For the fifth image(class=14), the model is misjudged that this is Traffic signals sign with not so much probability (probability of 0.2088), actually the image is stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .2088         		| Traffic signals   							| 
| .1687     			| General caution 								|
| .0808					| Stop											|
| .0704	      			| Speed limit (30km/h)					 		|
| .0461				    | No vehicles									|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


