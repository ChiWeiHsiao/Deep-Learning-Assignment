import os, random
import sys
import pickle
import numpy as np

def load_data():
    MNIST_M = np.load('/home/jay/DSN/Dataset/Mnist_Mnist_M/Mnist_M_rgb2.npy')
    train_data, train_label = MNIST_M[0]
    valid_data, valid_label = MNIST_M[1]
    test_data, test_label = MNIST_M[2]
    
    return train_data, train_label, valid_data,valid_label, test_data,test_label