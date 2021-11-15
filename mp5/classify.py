# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""
import math
import queue

import numpy as np


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    length = len(train_set[0])
    b = 0
    W = np.zeros(length);
    time = 0
    while time < max_iter:
        for i in range(len(train_set)):
            if train_labels[i] == 0:
                y = -1
            else:
                y = 1
            if y * (np.sum(W * train_set[i]) + b) <= 0:
                W = W + learning_rate * (y) * train_set[i]
                b = b + learning_rate * (y)
        time += 1

    return W, b


def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    para = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    ans = np.zeros(len(dev_set))
    for i in range(len(dev_set)):
        if np.sign(para[1] + np.sum(dev_set[i] * para[0])) <= 0:
            ans[i] = 0
        else:
            ans[i] = 1
    return list(ans)


def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    ans = np.zeros(len(dev_set))
    for i in range(len(dev_set)):
        pq = queue.PriorityQueue()
        isani = 0
        notani = 0
        for j in range(len(train_set)):
            pq.put((np.sqrt(np.sum(np.power(dev_set[i] - train_set[j],2))), train_labels[j]))
        for time in range(k):
            top = pq.get()[1]
            if not top:
                notani += 1
            else:
                isani += 1
        if notani < isani:
            ans[i] = 1
        else:
            ans[i] = 0
    return list(ans)
