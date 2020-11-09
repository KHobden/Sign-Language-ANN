#Sign Language
#Kieran Hobden
#07-Nov-'20

#This project aims to predict the number from a picture of its sign
#We will use an Artificial Neural Network (ANN) to make our predictions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

#Load the datasets
#X contains 2062 images of size 64x64 showing different pictures of different signs
#Y contains 2062 arrays of ints showing which number is being signed
#e.g. [0,1,0,0,0,0,0,0,0,0] shows a one is present
#whilst [1,0,0,0,0,0,0,0,0,0] shows a zero is present
X = np.load('input/X.npy')
Y = np.load('input/Y.npy')

#Check the dataset by observing two images
#Image 260 shows a zero and image 900 shows a one
plt.subplot(1,2,1)
plt.imshow(X[260])
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow((X[900]))
plt.axis('off')
plt.show() #Remove this line to not show the images

#For our purposes we will make a smaller dataset denoted by x and y to represent just zeros and ones
#All zeros occur between indices 204 and 409 and all ones occur between indices 822 and 1027
x = np.concatenate((X[204:409], X[822:1027]))
y = np.concatenate((np.zeros(205), np.ones(205))) #.reshape(x.shape[0],1)
#x is now a tensor of dimensions 410x64x64 which contains 410 images of size 64x64
#y is a vector of dimension 410 containing only a zero or one

#We want to test or model on 15% of our dataset so we can train it on the other 85%
#The random_state allows us to obtain reproducible results
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=3)
#x_train and y_train now contain 348 images and classifiers respectively
#x_test and y_test contain 62 images and classifiers respectively

#As we will not be using a Convolutional Neural Network (CNN), we must flatten our tensors to make them 2D
#x_train and x_test will have rows corresponding to images and 4096 columns containing the pixels in each image
#y_train and y_test will have one column with rows corresponding to each image classifier
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

#Build the Artificial Neural Network
def build_classifier():
	#Create a sequential model so we can add our layers in a simple, structured way
	classifier = Sequential([
		layers.Dense(units=64, kernel_initializer='uniform', activation='relu', input_dim=x_train.shape[1]),
		layers.Dense(units=16, kernel_initializer='uniform', activation='relu'),
		layers.Dense(units=4, kernel_initializer='uniform', activation='relu'),
		#Next layer has units=1 so that our model outputs a scalar
		#A sigmoid activation function is used to produce an output that lies between 0 and 1
		layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
	])

	#We use a binary crossentropy function for our loss as we are producing a binary result (0 or 1)
	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return classifier

classifier = KerasClassifier(build_fn=build_classifier, epochs=100, verbose=0)

#Calculate the accuracy of the model using cross validation
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
print("Accuracy mean:", str(accuracies.mean()))
print("Accuracy variance:", str(accuracies.std()))

#If the accuracy is sufficient, fit the model to our training data and use this to make predictions about our test data
classifier.fit(x_train, y_train, verbose=0)
preds = classifier.predict(x_test)

#Compute the error found when we compare our predictions to the true data
print("Mean Absolute Error of Test Dataset:", mae(y_test, preds))