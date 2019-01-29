################################################################################
# Author: Tin Nguyen, Samory Kpotufe. 
# Name: 
#
# Feature: 
# 
# Usage guideline:
# -
# 
# Dependencies: dyadic_pruning.py 
################################################################################

import numpy as np 
import sys 
import scipy.io.arff as arff
from sklearn import preprocessing
import sklearn.datasets as skd
import dyadic_pruning as pruning

# parse command-line arguments 
ex_index = int(sys.argv[1])
train_index = int(sys.argv[2])
penalty_type = str(sys.argv[3])
dataset_name = str(sys.argv[4])

classifer_name = "dyadic_pruning_" + penalty_type

# different datasets are loaded slightly differently 
if (dataset_name == "eeg"):
    eeg = arff.loadarff('realData/EEG Eye State.arff')
    eeg_dataset = np.asarray(eeg[0])
    data = np.zeros((len(eeg_dataset),14))
    target = np.zeros((len(eeg_dataset)))
    for i in xrange(len(eeg_dataset)):
        for j in xrange(14):
            data[i,j] = eeg_dataset[i][j]
        target[i] = eeg_dataset[i][14]
    target = target.astype(int)

elif (dataset_name == "epileptic"):
    seizure_dataset = np.genfromtxt('realData/epileptic_seizure_data.csv', delimiter=",", skip_header=1)
    data = seizure_dataset[:,1:179]
    target = seizure_dataset[:,179] > 1.1 # binarize labels 
    target = target.astype(int)

elif (dataset_name == "spam"):
    spam_dataset = np.loadtxt('realData/spambase.data',delimiter=",")
    data = spam_dataset[:,0:57]
    target = spam_dataset[:,57]

elif (dataset_name == "diagnosis"):
    diagnosis_dataset = np.genfromtxt('realData/Sensorless_drive_diagnosis.txt', delimiter=" ")
    ndim = len(diagnosis_dataset[0])
    data = np.zeros((len(diagnosis_dataset),ndim-1))
    target = np.zeros((len(diagnosis_dataset)))
    for i in xrange(len(diagnosis_dataset)):
        target[i] = diagnosis_dataset[i][ndim-1].astype(int) # last feature is label 
        for j in xrange(ndim-1):
            data[i,j] = diagnosis_dataset[i][j]
    target = target.astype(int)

elif (dataset_name == "digit"):
    digit_train = np.genfromtxt('realData/optdigits.tra', delimiter=",")
    train_data = np.zeros((len(digit_train),64))
    train_target = np.zeros((len(digit_train)))
    for i in xrange(len(digit_train)):
        train_target[i] = digit_train[i][64].astype(int)
        for j in xrange(64):
            train_data[i,j] = digit_train[i][j]
            
    digit_test = np.genfromtxt('realData/optdigits.tes', delimiter=",")
    test_data = np.zeros((len(digit_test),64))
    test_target = np.zeros((len(digit_test)))
    for i in xrange(len(digit_test)):
        test_target[i] = digit_test[i][64].astype(int)
        for j in xrange(64):
            test_data[i,j] = digit_test[i][j]

    data = np.concatenate((train_data,test_data),axis=0)
    target = np.concatenate((train_target,test_target),axis=0).astype(int)

elif (dataset_name == "crowd"):
    def crowd_label(class_name):
        if (class_name == "forest"):
            return 0
        elif (class_name == "grass"):
            return 1
        elif (class_name == "farm"): 
            return 2
        elif (class_name == "orchard"):
            return 3
        elif (class_name == "water"):
            return 4
        elif (class_name == "impervious"):
            return 5 
    crowd_train = np.genfromtxt('realData/crowd_training.csv',delimiter=',',skip_header=1, converters={0:crowd_label})
    data_train = np.zeros((len(crowd_train),28))
    target_train = np.zeros((len(crowd_train)))
    for i in xrange(len(crowd_train)):
        target_train[i] = crowd_train[i][0].astype(int)
        for j in xrange(28):
            data_train[i,j] = crowd_train[i][j+1]
    crowd_test = np.genfromtxt('realData/crowd_testing.csv',delimiter=',',skip_header=1, converters={0:crowd_label})
    data_test = np.zeros((len(crowd_test),28))
    target_test = np.zeros((len(crowd_test)))
    for i in xrange(len(crowd_test)):
        target_test[i] = crowd_test[i][0].astype(int)
        for j in xrange(28):
            data_test[i,j] = crowd_train[i][j+1]
    data = np.concatenate((data_train,data_test),axis=0)
    target = np.concatenate((target_train,target_test),axis=0)

elif (dataset_name == "wine"):
    def wine_label(class_name):
        return int(class_name)

    redwine_dataset = np.genfromtxt('realData/winequality-red.csv', delimiter=";", skip_header=1, converters={11:wine_label})
    red_data = np.zeros((len(redwine_dataset),11))
    red_target = np.zeros((len(redwine_dataset)))
    for i in xrange(len(redwine_dataset)):
        red_target[i] = redwine_dataset[i][11].astype(int)
        for j in xrange(11):
            red_data[i,j] = redwine_dataset[i][j]
            
    whitewine_dataset = np.genfromtxt('realData/winequality-white.csv', delimiter=";", skip_header=1, converters={11:wine_label})
    white_data = np.zeros((len(whitewine_dataset),11))
    white_target = np.zeros((len(whitewine_dataset)))
    for i in xrange(len(whitewine_dataset)):
        white_target[i] = whitewine_dataset[i][11].astype(int)
        for j in xrange(11):
            white_data[i,j] = whitewine_dataset[i][j]
            
    data = np.concatenate((red_data,white_data),axis=0)
    target = np.concatenate((red_target,white_target),axis=0).astype(int)

elif (dataset_name == "letter"):
    uppercase_alphabet = ["A","B", "C", "D","E", "F", "G", "H", "I", "J", "K", "L","M","N","O","P", "Q", "R", "S","T","U","V","W", "X", "Y","Z"]
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(uppercase_alphabet)
    def letter_label(class_name):
        return labelEncoder.transform([class_name])[0]

    letter_dataset = np.genfromtxt('realData/letter-recognition.data', delimiter=",",converters={0:letter_label})
    ndim = len(letter_dataset[0])
    data = np.zeros((len(letter_dataset),ndim-1))
    target = np.zeros((len(letter_dataset)))
    for i in xrange(len(letter_dataset)):
        target[i] = letter_dataset[i][0].astype(int) # last feature is label 
        for j in xrange(1,ndim):
            data[i,j-1] = letter_dataset[i][j]
    target = target.astype(int)

elif (dataset_name == "gas"):
    files = ["realData/Gas/batch"+str(i)+".dat" for i in xrange(1,11)]
    batches = skd.load_svmlight_files(files)

    data = batches[0].todense()
    target = batches[1]
    len_target = len(target)
    target = np.reshape(target,(len_target,1))

    for idx in xrange(2,11):
        batch_data = batches[(idx-1)*2].todense()
        batch_target = batches[2*idx-1]
        len_batch_target = len(batch_target)
        batch_target = np.reshape(batch_target,(len_batch_target,1))
        
        data = np.concatenate((data,batch_data),axis=0)
        target = np.concatenate((target, batch_target),axis=0)

    target = target.astype(int)
    total_size = data.shape[0]
    target = np.reshape(target, (total_size,))

total_size = data.shape[0]
test_size = 2000
if (dataset_name == "diagnosis"):
    # sample_size = np.linspace(10000,15000,num=5)
    sample_size = np.linspace(7000,10000,num=5)
elif (dataset_name == "gas"):
    sample_size = np.linspace((total_size-test_size)/2,total_size-test_size,num=5)
    # sample_size = np.linspace(3000, 6000, num=5)
else:
    sample_size = np.linspace((total_size-test_size)/2,total_size-test_size,num=5)
sample_size = sample_size.astype(int)

X_train, y_train, X_test, y_test, numOfLabels = pruning.preprocess(data,target,test_size=test_size,train_size=sample_size[train_index],random=ex_index)
# Fit, predict, report classification error 
classifier = pruning.DDT(numOfLabels, penalty_type)
classifier.fit(X_train, y_train)
testErr = 1-classifier.score(X_test, y_test)
filename = './results/accuracy/' + dataset_name + '-' + classifer_name + '-ex'+str(ex_index)+'tr'+str(train_index)+'.npy'
np.save(filename,testErr)

# report margin distribution 
if (train_index == 4):
    _, misClassifiedVector, margin_list = classifier.margin_distribution_for_abstain(X_test,y_test)
    margin_list = np.asarray(margin_list,dtype=float)

    filename = './results/margin/' + dataset_name + '-' + classifer_name + '-ex' + str(ex_index)+'tr'+str(train_index)+'.npy'
    margins = np.concatenate((misClassifiedVector.reshape(test_size,1), margin_list.reshape(test_size,1)), axis=1)
    np.save(filename,margins)