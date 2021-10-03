# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir, testdir, stemming, lowercase,
                                                                       silently)
    return train_set, train_labels, dev_set, dev_labels


# Keep this in the provided template
def print_paramter_vals(laplace, pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""


def naiveBayes(train_set, train_labels, dev_set, laplace=0.5, pos_prior=0.5, silently=False):
    # Keep this in the provided template
    print_paramter_vals(laplace, pos_prior)
    positve = []
    negative = []
    n_positive = 0
    n_negative = 0
    for i in range(train_labels.size):
        if train_labels[i] == 1:
            n_positive += len(train_set[i])
            positve.extend(train_set[i])
        else:
            n_negative += len(train_set[i])
            negative.extend(train_set[i])

    positve = Counter(positve)
    negative = Counter(negative)
    s = set(positve.keys())
    s = s | set(negative.keys())
    v = len(s)  # number of word TYPES seen in training data
    unk_posi = math.log(laplace / (n_positive + laplace * (1 + v)))
    unk_nega = math.log(laplace / (n_negative + laplace * (1 + v)))
    for key in positve.keys():
        positve[key] = math.log((positve[key] + laplace) / (n_positive + laplace * (1 + v)))
    for key in negative.keys():
        negative[key] = math.log((negative[key] + laplace) / (n_negative + laplace * (1 + v)))

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        nega_pob = math.log(1 - pos_prior)
        posi_pob = math.log(pos_prior)
        for word in doc:
            if word in positve.keys():
                posi_pob += positve[word]
            else:
                posi_pob += unk_posi
            if word in negative.keys():
                nega_pob += negative[word]
            else:
                nega_pob += unk_nega

        if posi_pob > nega_pob:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.37,
                pos_prior=0.5, silently=False, n_positive=None):
    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)
    bi_positive = {}
    bi_negative = {}
    positive = {}
    negative = {}
    n_positive = 0
    n_negative = 0
    n_bipositive = 0
    n_binegative = 0
    for i in range(len(train_set)):
        if train_labels[i] == 0:
            n_negative += len(train_set[i])
            n_binegative += len(train_set[i])-1
            for j in range(len(train_set[i])):
                if train_set[i][j] in negative.keys():
                    negative[train_set[i][j]] += 1
                else:
                    negative[train_set[i][j]] = 1
                if j < len(train_set[i]) - 1:
                    if train_set[i][j] + train_set[i][j + 1] in bi_negative.keys():
                        bi_negative[train_set[i][j] + train_set[i][j + 1]] += 1
                    else:
                        bi_negative[train_set[i][j] + train_set[i][j + 1]] = 1
        else:
            n_positive += len(train_set[i])
            n_bipositive += len(train_set[i]) - 1
            for j in range(len(train_set[i])):
                if train_set[i][j] in positive.keys():
                    positive[train_set[i][j]] += 1
                else:
                    positive[train_set[i][j]] = 1
                if j < len(train_set[i]) - 1:
                    if train_set[i][j] + train_set[i][j + 1] in bi_positive.keys():
                        bi_positive[train_set[i][j] + train_set[i][j + 1]] += 1
                    else:
                        bi_positive[train_set[i][j] + train_set[i][j + 1]] = 1

    s = set(positive.keys())
    s = s | set(negative.keys())
    v = len(s)  # number of word TYPES seen in training data
    s = set(bi_positive.keys())
    s = s | set(bi_negative.keys())
    v_bi = len(s)  # number of bi_word TYPES seen in training data

    unk_posi = math.log(unigram_laplace / (n_positive + unigram_laplace * (1 + v)))
    unk_nega = math.log(unigram_laplace / (n_negative + unigram_laplace * (1 + v)))
    unk_biposi = math.log(bigram_laplace / (n_bipositive + bigram_laplace * (1 + v_bi)))
    unk_binega = math.log(bigram_laplace / (n_binegative + bigram_laplace * (1 + v_bi)))

    for key in positive.keys():
        positive[key] = math.log((positive[key] + unigram_laplace) / (n_positive + unigram_laplace * (1 + v)))
    for key in negative.keys():
        negative[key] = math.log((negative[key] + unigram_laplace) / (n_negative + unigram_laplace * (1 + v)))
    for key in bi_positive.keys():
        bi_positive[key] = math.log((bi_positive[key] + bigram_laplace) / (n_bipositive + bigram_laplace * (1 + v_bi)))
    for key in bi_negative.keys():
        bi_negative[key] = math.log((bi_negative[key] + bigram_laplace) / (n_binegative + bigram_laplace * (1 + v_bi)))

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        binega_pob = nega_pob = math.log(1 - pos_prior)
        biposi_pob = posi_pob = math.log(pos_prior)
        for i in range(len(doc)):
            if doc[i] in positive.keys():
                posi_pob += positive[doc[i]]
            else:
                posi_pob += unk_posi
            if doc[i] in negative.keys():
                nega_pob += negative[doc[i]]
            else:
                nega_pob += unk_nega
            if i < len(doc)-1:
                if doc[i]+doc[i+1] in bi_positive.keys():
                    biposi_pob += bi_positive[doc[i]+doc[i+1]]
                else:
                    biposi_pob += unk_biposi
                if doc[i]+doc[i+1] in bi_negative.keys():
                    binega_pob += bi_negative[doc[i]+doc[i+1]]
                else:
                    binega_pob += unk_binega
        if ((1-bigram_lambda)*posi_pob+bigram_lambda*biposi_pob)>((1-bigram_lambda)*nega_pob+bigram_lambda*binega_pob):
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats
