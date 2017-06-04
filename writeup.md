#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample/vis1.png "vis1"
[image2]: ./sample/vis2.png "vis2"
[image3]: ./sample/vis3.png "vis3"
[image4]: ./sample/vis4.png "vis4"
[image5]: ./sample/0.ppm "vis5"
[image6]: ./sample/1.ppm "vis6"
[image7]: ./sample/2.ppm "vis7"
[image8]: ./sample/3.ppm "vis8"
[image9]: ./sample/4.ppm "vis9"
[image10]: ./sample/vis10.png "vis10"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

Python's 'len' and 'shape' are used to find the number of training examples and numpy's 'unique' is used to find the number of classes in the training set. 

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is distributed. As one can see, the original training data is very uneven.

![Original Training Set][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth code cell of the IPython notebook.

As a first step, images are converted to grayscale and normalized to make use of the original Lenet Project with its one channel input and minimize data differences among the sample pictures.  

Here are examplse of a traffic sign images before preprocessing.

![Traffic Signs][image2]

Once the images are grayscaled and normalized, the images are jittered to artificially augment the given dataset. Three properties are used when jittering the images: position, scale, and rotation (Pierre Sermanet and Yann LeCun artical's parameters were taken into consideration). To have an evenly distributed dataset, the number of images for each class is taken into consideration when calculating how many jittered images are needed.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Since the data was already split into training, validation, and test, there was no need for splitting. The data augmentation is mentioned above and is done to have each class trained equally. Skewed training dataset can produce uneven correctness in result.

To have evenly distributed training data, the total number of jittered images that need to be created is calculated by subtracting user-set training data value (3000) - the total number of images in each class.

Here is a graph of how many jittered images were created.
![jittered graph][image3]

Here are samples of jittered images
![jittered images][image4]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the twelveth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x100 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x120 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x120 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x180 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x200 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x200 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 4x4x200     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x200 			     	|
| Flatten				| 800        									|
| Dropout	            | 0.5											|
| Fully connected		| 80											|
| Softmax				| 43        									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventeenth cell of the ipython notebook. 
Other than the addition of dropout, the same procedure as Lenet project is used.

For training the model, these parameters are used:
Type of optimizer: Adam Optimizer
Batch size: 128
Number of epochs: 30
Learning rate: 0.001
Dropout rate: 0.5


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.990
* test set accuracy of 0.983

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Iterative approach was taken for this project. At first, the original Lenet structure was used as the model. However, the original Lenet model used only produced validation accuracy in high 80's. The Lenet model was believed to be too shallow and thin for this project, so depth and width were added to the model to compensate for the complexity of the data. two more convolution layers were added with pooling every two layers and dropout was added towards the end to decrease the chance of overfitting. The number of epoch was originally 25, but the accuracy seemed to stay near the same range after 7~10 epochs so at first, it was decreased down to 15. After some repetitive adjustments, it was increased to 30 to make sure to the highest possible validation set accuracy. The recommended learning rate was used with the original batch size as the Lenet model. 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Sign1][image5] ![Sign2][image6] ![Sign3][image7] ![Sign4][image8] ![Sign5][image9]

The first image might be difficult to classify because the traffic sign is under another traffic sign that is partly shown.
The second image might be difficult to classify because the pole that the sign is attatched to takes a big portion of the image.
The third image might be difficult to classify because like the first image, the intended sign is below another sign that is partly shown.
The fourth image might be difficult to classify because the image contains a white line (seems to be a part of a structure) that throw the model off.
The fifth image might be difficult to classify because the image is very dark, it is very hard to see.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twenty second cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (20km/h)  | Speed Limit (20km/h)  						| 
| Speed Limit (50km/h)  | Speed Limit (50km/h)							|
| Priority Road			| Priority Road									|
| Keep Right	      	| Keep Right					 				|
| Roundabout			| Roundabout     								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 98.3%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a Speed Limit (20km/h) sign(probability of 1.0), and the image does contain a Speed Limit (20km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed Limit (20km/h)   						| 
| 0.00     				| Speed Limit (120km/h)							|
| 0.00 					| Speed Limit (30km/h)							|
| 0.00 	      			| Wild Animals Crossing							|
| 0.00 				    | Slippery Road     							|


For the second image, the model is absolutely sure that this is a Speed Limit (50km/h) sign(probability of 1.0), and the image does contain a Speed Limit (50km/h) sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed Limit (50km/h)   						| 
| 0.00     				| Speed Limit (60km/h)							|
| 0.00 					| Speed Limit (80km/h)							|
| 0.00 	      			| Speed Limit (30km/h)							|
| 0.00 				    | Yield      									|


For the third image, the model is absolutely sure that this is a Priority Roead sign(probability of 1.0), and the image does contain a Priority Road sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority Road 		  						| 
| 0.00     				| Keep Right									|
| 0.00 					| End of All Speed and Passing Limits			|
| 0.00 	      			| Roundabout									|
| 0.00 				    | No Vehicles   	  							|


For the fourth image, the model is absolutely sure that this is a Keep Right sign(probability of 1.0), and the image does contain a Keep Right sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep Right			  						| 
| 0.00     				| Priority Road									|
| 0.00 					| Turn Left Ahead								|
| 0.00 	      			| Speed Limit (80km/h)							|
| 0.00 				    | End of Passing by Vehicles Over 3.5 Tons   	|

For the last image, the model is absolutely sure that this is a Roundabout sign(probability of 1.0), and the image does contain a Roundabout sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Roundabout			   						| 
| 0.00     				| Speed Limit (30km/h)							|
| 0.00 					| Priority Road									|
| 0.00 	      			| Keep Right									|
| 0.00 				    | Go Straight or Left      						|

![Soft Max Probabilities][image10]