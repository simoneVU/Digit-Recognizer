from re import T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn


def init_params():
    #Randomly initialize the weight matrices (and biases) for the layers of the NN
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

def ReLU(Z):
    #maximum() returns 0 if the value is smaller than 0 or 1 otherwise.
    return np.maximum(0,Z)

def softmax(Z):
    #print("Z is :" + str(Z))
    return np.exp(Z)/np.sum(np.exp(Z),axis=0)
    
def forward(W1, b1, W2, b2, input_matrix):
    Z1 = W1.dot(input_matrix) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(np.array(Z2, dtype=np.float128))
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    #For each row, go through the column specified by the label in Y and set it to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    #Transpose to get each column as example
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0 # TRUE = 1, FALSE = 0.

def backward(Z1, A1, Z2, A2, W2, X, Y):
    #First one-hot encode Y, turn the labels into matrix
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y #Mean Swuare error derivative

    #Other partial derivatives
    dW2 = 1/Y.size * dZ2.dot(A1.T)
    db2 =  1/Y.size * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/Y.size * dZ1.dot(X.T)
    db1 =  1/Y.size * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    #Vanilla update of the network parameters
    W1 =  W1 - lr * dW1
    b1 =  b1 - lr * db1
    W2 =  W2 - lr * dW2
    b2 =  b2 - lr * db2
    #print(W1, b1, W2, b2)    
    return W1, b1, W2, b2

def get_predictions(A2):
    #Get the digit with the highest prediction value.
    # A2 is a (10, 60000) matrix.
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, lr):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 =  forward(W1, b1, W2, b2, X)
        #print("A2 is :" + str(A2))
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
            
    return W1, b1, W2, b2