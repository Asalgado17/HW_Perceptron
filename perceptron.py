import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import random

path_data = 'perceptron.csv'

perceptron_raw_data = pd.read_csv(path_data, delimiter=",")

perceptron_raw_data

# changing to labels L or T to numbers
perceptron_raw_data.LT = perceptron_raw_data.LT.map({'T': 1, 'L': -1})

def activation_function(z):
    return np.where(z >= 0, 1, -1)

def fit(X_train, y_train):
    weights = np.zeros(16)
    bias = 0
    learning_rate = 0.01

    for _ in range(1000):
        for idx, x_i in enumerate(X_train):
            linear_product = np.dot(x_i, weights) + bias    
            y_pred = activation_function(linear_product)

            update = learning_rate * (y_pred - y_train[idx]) 

            weights = weights - update * x_i 
            bias = bias - update

    return weights, bias

X = perceptron_raw_data.iloc[:, :16].values  # Features
y = perceptron_raw_data.iloc[:, 16].values  # Labels L or T

X.shape
y.shape
y

random_seed = int(random.random() * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)
print("y_train shape: ",y_train.shape)
print("y_test shape: ",y_test.shape)

weights, bias = fit(X_train, y_train)
print("Weights: ",weights)
print("Bias: ",bias)

def predict(X_test, weights, bias):
    # x * w
    linear_product = np.dot(X_test, weights) + bias  # y = w * x + b
    y_pred = activation_function(linear_product)
    return y_pred

y_pred = predict(X_test, weights, bias)
y_pred
y_test

def print_stats_performance_metrics(y_test, y_pred):
    
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    print('Precision: %.2f' % precision_score(y_test, y_pred, average='weighted'))
    print('Recall: %.2f' % recall_score(y_test, y_pred, average='weighted'))
    print('F1_score: %.2f' % f1_score(y_test, y_pred, average='weighted'))

print_stats_performance_metrics(y_test, y_pred)


np.save('weights.npy', weights)
np.save('bias.npy', bias)