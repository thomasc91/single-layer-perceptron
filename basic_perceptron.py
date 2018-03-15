# -*- coding: utf-8 -*-
"""
Code adapted from https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
"""


from random import randrange
from csv import reader
import plotly.plotly as py
import pandas as pd
from plotly.graph_objs import *


#Load CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

#Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

#Split dataset into k folds (a fold is an equal sized subsample of the data)
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(0, len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split        

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
# k fold cross validation splits the data into k subsamples.
# Each subsample is used to test the data, while the remaining k-1 samples are used to train the data.
# The process is repeated k times, and the k results are averaged to produce a single estimation
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold) #size of training set becomes k-1
		train_set = sum(train_set, []) 
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

#Predicts an output value for a row given a set of weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation>= 0.0 else 0.0

#Training the weights for our perceptron using stochastic gradient descent
"""
Stochastic gradient descent requires two parameters:
    Learning rate: Limits the amount each weight is corrected when the weights are updated
    Epochs: The number of times to run through the training data while updating the weight
"""

def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))] #one weight for each input attribute
    for epoch in range(n_epoch): #loop over each epoch
        sum_error = 0.0
        for row in train: #loop over each row of the training data 
            prediction = predict(row, weights)
            error = row[-1] - prediction #measure the difference between the expected and the predicted values
            sum_error += error**2 #take the sum square of the error per epoch
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1): #loop over each weight and update for a row in each epoch
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        print((epoch, sum_error))
        error_graph_data.append((epoch, sum_error))
    return weights
        
# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = predict(row, weights)
		predictions.append(prediction)
	return(predictions)
    
# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 2
l_rate = 0.02
n_epoch = 1000

error_graph_data = list()
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)

df = pd.DataFrame(error_graph_data, columns=['epoch', 'error'])

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
