################################################################################
# Author: Tin Nguyen, Samory Kpotufe. 
# Name:  dataMaster_[dyadic_pacbayes]_[eTS,MD].py 
#
# Feature: 
# 
# Usage guideline:
# -
# Future work:
# - instead of using Decimal class to store the messages alpha and beta, does it 
# make sense to normalize them along the dynamic program?  
# 
# Dependencies: dyadic_pacbayes.py 
################################################################################

import numpy as np 
import sys 
import scipy.io.arff as arff
from sklearn import preprocessing
import sklearn.datasets as skd
import dyadic_pacbayes as aggregate

# parse command-line arguments 
ex_index = int(sys.argv[1])
train_index = int(sys.argv[2])
dataset_name = str(sys.argv[3])

classifer_name = "dyadic_pacbayes"

# different datasets are loaded slightly differently 
if (dataset_name == "spam"):
    spam_dataset = np.loadtxt('../datasets/spam/spambase.data',delimiter=",")
    data = spam_dataset[:,0:57]
    target = spam_dataset[:,57]

elif (dataset_name == "digit"):
    digit_train = np.genfromtxt('../datasets/digit/optdigits.tra', delimiter=",")
    train_data = np.zeros((len(digit_train),64))
    train_target = np.zeros((len(digit_train)))
    for i in xrange(len(digit_train)):
        train_target[i] = digit_train[i][64].astype(int)
        for j in xrange(64):
            train_data[i,j] = digit_train[i][j]

# allocate training and testing sets 
total_size = data.shape[0]
test_size = 2000
if (dataset_name == "diagnosis"):
    # sample_size = np.linspace(10000,15000,num=5)
    sample_size = np.linspace(7000, 10000, num=5)
elif (dataset_name == "gas"):
    sample_size = np.linspace((total_size-test_size)/2,total_size-test_size,num=5)
    # sample_size = np.linspace(3000, 6000, num=5)
else:
    sample_size = np.linspace((total_size-test_size)/2,total_size-test_size,num=5)
sample_size = sample_size.astype(int)

X_train, y_train, X_test, y_test, numOfLabels = aggregate.preprocess(data,target,test_size=test_size,train_size=sample_size[train_index],random=ex_index)
# Fit, predict, report classification error 
classifier = aggregate.DDT(numOfLabels)
classifier.fit(X_train, y_train)
testErr = 1-classifier.score(X_test, y_test)
filename = '../results/accuracy/' + dataset_name + '-' + classifer_name + '-ex'+str(ex_index)+'tr'+str(train_index)+'.npy'
np.save(filename,testErr)

# report margin distribution 
if (train_index == 4):
    # log ratio margin 
    _, misClassifiedVector, margin_list = classifier.margin_distribution(X_test,y_test,"log ratio")
    margin_list = np.asarray(margin_list,dtype=float)
    max_margin = np.max(margin_list[margin_list != np.inf])
    margin_list[margin_list == np.inf] = 1.5*max_margin # some really large number 

    filename = '../results/margin/' + dataset_name + '-' + classifer_name + "-margin=log ratio" +  '-ex' + str(ex_index)+'tr'+str(train_index)+'.npy'
    margins = np.concatenate((misClassifiedVector.reshape(test_size,1), margin_list.reshape(test_size,1)), axis=1)
    np.save(filename,margins)

    # average gap margin 
    _, misClassifiedVector, margin_list = classifier.margin_distribution(X_test,y_test,"average gap")
    margin_list = np.asarray(margin_list,dtype=float)

    filename = '../results/margin/' + dataset_name + '-' + classifer_name + "-margin=average gap" +  '-ex' + str(ex_index)+'tr'+str(train_index)+'.npy'
    margins = np.concatenate((misClassifiedVector.reshape(test_size,1), margin_list.reshape(test_size,1)), axis=1)
    np.save(filename,margins)