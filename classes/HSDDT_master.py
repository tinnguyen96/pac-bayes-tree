################################################################################
# Author:    Tin Nguyen
# Name:  HSDDT_master.py 
# Most recent update: 04/19/2018
#
# Differences compared to AkdDT_nS_master.py 
# - include maximal depth and collapse of hi-lo as terminating conditions 
# - in computing exponential of phi, use Decimal instead of np.int64 for best possible
# precision 
# - include check for correctness of growth stage 
# Dependencies on other modules in the project: Node.py
################################################################################
import numpy as np
import Queue 
import time
from decimal import * 
import Node_A as Node
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def split(repArray, axis, isLeft):
    newRepArray = np.copy(repArray)
    newRepArray[axis,1] += 1
    newRepArray[axis,0] *= 2
    if (not isLeft):
        newRepArray[axis,0] += 1
    return newRepArray

def preprocess(data,target,test_size,train_size=None,random=42):
    # standardize train set to live in unit hypercube 
    minMaxScaler = preprocessing.MinMaxScaler()
    data = minMaxScaler.fit_transform(data)

    # standardize classification labels 
    labelEncoder = preprocessing.LabelEncoder()
    labelEncoder.fit(target)

    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=train_size,test_size=test_size, random_state=random)

    y_train = labelEncoder.transform(y_train)
    y_test = labelEncoder.transform(y_test)
    numOfLabels = len(labelEncoder.classes_)

    return (X_train, y_train, X_test, y_test, numOfLabels)

def default_updateFunction(r, beta):
    return Decimal(1) - (Decimal(1) - beta)*r

def default_getC_predictFunction(beta):
    one_decimal = Decimal(1)
    one_half_decimal = Decimal(0.5)
    temp = one_decimal.fma(2,0)/(beta+one_decimal)
    c = (one_decimal+beta)*temp.ln()/((one_decimal-beta)*one_decimal.fma(2,0))
    return c

def default_predictFunction(r, beta):
    one_decimal = Decimal(1)
    one_half_decimal = Decimal(0.5)
    temp = one_decimal.fma(2,0)/(beta+one_decimal)
    c = (one_decimal+beta)*temp.ln()/((one_decimal-beta)*one_decimal.fma(2,0))
    if (r <= one_half_decimal-c):
        return 0
    elif (r >= one_half_decimal+c):
        return 1 
    else:
        return one_half_decimal-(one_decimal-r.fma(2,0))/c.fma(4,0)

class DDT:
    ## Input: 
    def __init__(self, numOfLabels, check_dyn_prog=False, one_shot=False, updateFunction=default_updateFunction, predictionFunction=default_predictFunction, val_ratio=0.35, 
                lambda_range=map(Decimal, np.logspace(-8, 0.95, num=10, base=2.0)), linear_search=True, traverse=False,
                check_growth=False, report_lambda=False):
        self._leaves, self._alter_leaves = Queue.PriorityQueue(), Queue.PriorityQueue()
        self._lambda_range, self._lambda_count = lambda_range, len(lambda_range)
        self._numOfLabels = numOfLabels
        self._updateFunction = updateFunction
        self._predictionFunction = predictionFunction
        self._linear_search = linear_search
        self._val_ratio = val_ratio
        self._traverse = traverse
        self._check_dyn_prog = check_dyn_prog
        self._one_shot = one_shot
        self._check_growth = check_growth
        self._report_lambda = report_lambda

    ## Input: X = data component of sample, y = label component
    def fit(self, X, y, random_state=42):
        # First split X into a train and validation component to determine best lambda paramaters 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self._val_ratio, random_state=random_state)
            
        # Instantiate instance variables 
        self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)
        self._X_val, self._y_val, self._valSize = X_val, y_val, len(y_val)
        self._dim = X_train.shape[1]
        self._max_splits = int(np.floor(np.log(self._trainSize)))

        # (?) what's the role of treeArr
        self._treeArr = range(self._trainSize)
        self._unitCube = np.zeros((self._dim,2),dtype=np.int)

        # construct template tree
        self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)
        if (self._check_growth):
            # print "Tree array at the end of growth procedure %s" %str(self._treeArr)
            # print "Training data is %s" %str(self._X_train)
            print "Check correctness of growth procedure: %s" %str(self.check_growth(self._root, 0, self._trainSize))

        print '-'*70  

        if (self._traverse):
            print "Traverse the template tree constructed using construction portion of training set"
            self.__traverse(self._root)
            print '-'*70

        if (self._one_shot):
            lambda_index = 0
            self.one_predict_one_update(X_val[0,:], y_val[0], lambda_index)
        else:
            # Search for best lambda using validation set 
            ## Evaluate performance of different parameters on validation set 
            self.__search_for_lambda()
            print '-'*70
            
            # After the best lambda parameters have been found, train the aggregate on the whole sample 
            X_train = np.concatenate((X_train,X_val),axis=0)
            y_train = np.append(y_train,y_val)
            self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)

            print '-'*70  
            print "Done with validation, now use whole training batch to construct template tree"
            self.__fullSample()

    def __search_for_lambda(self):
        print '-'*70
        print "Log search for lambda"
        print "For each parameter construct one weight scheme using the construction portion of training set"
        for lambda_index in xrange(self._lambda_count):
            _ = self.__get_loss(self._X_train, self._y_train, True, lambda_index)

        self._best_model_index, self._best_val_loss, self._val_loss = self.__optimize_lambda()
        self._best_lambda = self._lambda_range[self._best_model_index]

        if (self._report_lambda):
            print "Lambda range after log search for error-lambda %s" %str(self._lambda_range)
            print "Validation errors after log search for error-lambda %s" %str(self._val_loss)
            print "Value of best error-lambda %s" %str(self._best_lambda)

        if (self._linear_search):
            print '-'*70
            print "Reset weights"
            self.__reset(self._root)

            print "Linear search for lambda" 
            self._lambda_range = map(Decimal, np.linspace(float(self._best_lambda)/2,float(self._best_lambda)*2, num=self._lambda_count))
            print "For each parameter construct one weight scheme using the construction portion of training set"
            for lambda_index in xrange(self._lambda_count):
                _ = self.__get_loss(self._X_train, self._y_train, True, lambda_index)

            self._best_model_index, self._best_val_loss, self._val_loss = self.__optimize_lambda()
            self._best_lambda = self._lambda_range[self._best_model_index]

            if (self._report_lambda):
                print "Lambda range after linear search for error-lambda %s" %str(self._lambda_range)
                print "Validation errors after linear search for error-lambda %s" %str(self._val_loss)
                print "Value of best error-lambda %s" %str(self._best_lambda)

    def __fullSample(self):
        self.lambda_range = np.array([self._best_lambda])
        self._lambda_count = 1
        self._best_model_index = 0
        
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)
        self._unitCube = np.zeros((self._dim,2),dtype=np.int)

        # construct template tree
        self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)
        # use train set to fix weights
        print "Use train set to fix weights"
        self.__get_loss(self._X_train, self._y_train, return_average=True, lambda_index=None)

    ## method to update empirical error  
    def __update_regVal(self, node):
        trainInfo = node.get_trainInfo()
        exCount = sum(trainInfo)
        if (exCount > 0):
            regVal = trainInfo[1]/float(exCount)
        else:
            regVal = 0
        label = Decimal(regVal)
        node.set_label(label)

    def __grow(self, parent, repArray, axis, depth, lo, hi):
        labelCount = np.zeros(self._numOfLabels, dtype=np.int16)
        for i in xrange(lo,hi):
            exampleIdx = self._treeArr[i]
            y = self._y_train[exampleIdx]
            labelCount[y] += 1

        # Three terminating conditions: if the node contains only one label-class of points, no need to split further
        trueCount = np.asarray(labelCount > 0, dtype=np.int)
        if (hi - lo <= 1) or (np.sum(trueCount) == 1) or (depth == self._max_splits*self._dim):
            currNode = Node.Node(parent, None, axis, depth, self._lambda_count)
            self._leaves.put(currNode)
        else:
            # sort contiguous subset of treeArr from lo (inclusive) to hi (exclusive) in-place
            temp = self._treeArr[lo:hi] 
            temp.sort(key=lambda x: self._X_train[x,axis])
            self._treeArr[lo:hi] = temp

            # infer threshold 
            numOfHalves = depth/self._dim+1
            resolution = 2**(-numOfHalves)
            start,splits = repArray[axis,0], repArray[axis,1]
            lowerBound = start/float(np.power(2,splits))
            threshold = lowerBound + resolution

            # search for split location in treeArr, which is now sorted in the axis 
            leftLo, rightHi = lo, hi
            leftHi, rightLo = hi, hi
            for splitIdx in xrange(lo,hi):
                exampleIdx = self._treeArr[splitIdx]
                if (self._X_train[exampleIdx,axis] > threshold):
                    leftHi, rightLo = splitIdx, splitIdx
                    break

            # new rep arrays 
            leftRepArray = split(repArray, axis, True)
            rightRepArray = split(repArray, axis, False)

            # split according to the median
            currNode = Node.Node(parent, threshold, axis, depth, self._lambda_count)
            leftChild = self.__grow(currNode, leftRepArray, (axis+1)%self._dim, depth+1, leftLo, leftHi)
            currNode.set_left(leftChild)
            rightChild = self.__grow(currNode, rightRepArray, (axis+1)%self._dim, depth+1, rightLo, rightHi)
            currNode.set_right(rightChild)

        currNode.set_trainInfo(labelCount)
        self.__update_regVal(currNode)
        # use phi field to save weight
        currNode.set_phi(map(Decimal, np.ones(self._lambda_count)))
        # use beta field to save \bar{weight} 
        currNode.set_beta(map(Decimal, np.ones(self._lambda_count)))
        return currNode

    def __reset(self, node):
        node.set_phi(map(Decimal, np.ones(self._lambda_count)))
        node.set_beta(map(Decimal, np.ones(self._lambda_count)))
        if (not node.get_threshold() is None):
            self.__reset(node.get_left())
            self.__reset(node.get_right())

    # recursive function to compute prediction 
    def __predict_helper(self, point, currNode, onPath, model_index):
        regVal = currNode.get_label()
        threshold = currNode.get_threshold()

        if (onPath):
            if (threshold is None):
                wpred = currNode.get_phi()[model_index]*regVal
            else:
                axis = currNode.get_axis()
                if (point[axis] > threshold):
                    pointInRight, pointInLeft = True, False
                else:
                    pointInRight, pointInLeft = False, True
                wpred = self.__predict_helper(point, currNode.get_right(), pointInRight, model_index)*self.__predict_helper(point, currNode.get_left(), pointInLeft, model_index)
                wpred = (wpred + currNode.get_phi()[model_index]*regVal)/Decimal(2)
        else:
            wpred = currNode.get_beta()[model_index]
        return wpred 

    def predict(self, point,  lambda_index=None, get_weight=False):
        if lambda_index is None:
            model_index = self._best_model_index
        else:
            model_index = lambda_index
        wpred =  self.__predict_helper(point, self._root, True, model_index)
        normalized_wpred = wpred/self._root.get_beta()[model_index]
        output = self._predictionFunction(normalized_wpred, self._lambda_range[model_index])
        if (get_weight):
            return output, normalized_wpred
        else:
            return output

    def __update_helper(self, currNode, point, target, onPath, model_index):
        regVal = currNode.get_label()
        threshold = currNode.get_threshold()

        # update weight 
        if (onPath):
            past_weight = currNode.get_phi()
            new_weight = past_weight[model_index]*self._updateFunction(abs(regVal-target), self._lambda_range[model_index])
            past_weight[model_index] = new_weight
            currNode.set_phi(past_weight)

        # update \bar{weight}
        if (onPath):
            curr_weight = currNode.get_phi()
            if (threshold is None):
                currNode.set_beta(curr_weight)
                ans = curr_weight[model_index]
            else:
                axis = currNode.get_axis()
                if (point[axis] > threshold):
                    pointInRight, pointInLeft = True, False
                else:
                    pointInRight, pointInLeft = False, True
                temp = self.__update_helper(currNode.get_right(),point, target, pointInRight, model_index)*self.__update_helper(currNode.get_left(),point, target, pointInLeft, model_index)
                ans = (curr_weight[model_index] + temp)/Decimal(2)

                curr_bar_weight = currNode.get_beta()
                curr_bar_weight[model_index] = ans
                currNode.set_beta(curr_bar_weight)
        else:
            # no change to \bar{weight} if node off path 
            ans = currNode.get_beta()[model_index]
        return ans

    def update(self, point, target, lambda_index=None):
        if lambda_index is None:
            model_index = self._best_model_index
        else:
            model_index = lambda_index
        return self.__update_helper(self._root, point, target, True, model_index)

    def __get_loss(self, X_train, y_train, return_average=True, lambda_index=None):
        test_count = len(y_train)
        total_loss = 0
        for test_ex_index in xrange(test_count):
            point = X_train[test_ex_index,:]
            target = y_train[test_ex_index]
            output = self.predict(point, lambda_index)
            self.update(point, target, lambda_index)
            total_loss += float(abs(output-target))
        if (return_average):
            return total_loss/float(test_count)
        else:
            return total_loss

    def one_predict_one_update(self, point, target, lambda_index):
        print "Check dynamic program before any prediction or update"
        print self.check_dyn_prog(self._root, lambda_index)

        print "-"*70

        output = self.predict(point, lambda_index)
        root_bar_weight = self.update(point, target, lambda_index)

        print "after one prediction/update"
        print "new bar_weight at root %s" %str(root_bar_weight)
        print self.check_dyn_prog(self._root, lambda_index)

     ## Output: use the validation set to determine the best lambda parameter and the correspondinng 
    ## validation error of the aggregate 
    def __optimize_lambda(self):
        valLoss = np.zeros(self._lambda_count)
        for val_ex_index in xrange(self._valSize):
            point = self._X_val[val_ex_index,:]
            target = self._y_val[val_ex_index]
            for lambda_index in xrange(self._lambda_count):
                output = self.predict(point, lambda_index)
                if (output > 0.5):
                    predLabel = 1
                else:
                    predLabel = 0
                if predLabel != target:
                    valLoss += 1

        if (self._check_dyn_prog):
            for lambdaIdx in xrange(self._lambda_count):
                print "Check correctness of the dynamic program for the %s lambda value after validation process" %str(lambdaIdx)
                print self.check_dyn_prog(self._root, lambdaIdx)

        best_model_index = np.argmin(valLoss)
        best_val_loss = valLoss[best_model_index]
        return (best_model_index, best_val_loss, valLoss)

    def __traverse(self, node):
        print "axis %s" %str(node.get_axis())
        print "threshold %s" %str(node.get_threshold())
        if node.get_threshold() is None:
            print "Reached terminal leaf of depth %s" %str(node.get_depth())
            return 
        else:
            self.__traverse(node.get_left())
            self.__traverse(node.get_right())

    def check_dyn_prog(self, node, lambdaIdx, tol=1e-10):
        # print "axis %s" %str(node.get_axis())
        # print "threshold %s" %str(node.get_threshold())
        # print "train_info %s" %str(node.get_trainInfo())
        weight = node.get_phi()[lambdaIdx]
        # print "weight %s" %str(weight)
        bar_weight = node.get_beta()[lambdaIdx]
        
        if (node.get_threshold() is None):
            correctness = (abs(weight-bar_weight) < tol)
            return correctness
        else:
            left = node.get_left()
            right = node.get_right()
            
            bar_weight_left = left.get_beta()[lambdaIdx]
            bar_weight_right = right.get_beta()[lambdaIdx]
        
            # print "bar_weight stored %s" %str(bar_weight)
            # print "bar_weight according to dyn. prog. %s" %str((weight+bar_weight_left*bar_weight_right)/Decimal(2))

            correctness = (abs(bar_weight - (weight+bar_weight_left*bar_weight_right)/Decimal(2)) < tol)

            if not (correctness):
                return False
            else:
                return (self.check_dyn_prog(left, lambdaIdx) and self.check_dyn_prog(right, lambdaIdx))        

    def score(self, X_test, y_test, lambda_index=None):
        testCount = len(y_test)
        accuracyCount = 0
        for i in xrange(testCount):
            point = X_test[i,:]
            target = y_test[i]
            output = self.predict(point, lambda_index)
            if (output > 0.5):
                predLabel = 1
            else:
                predLabel = 0
            if predLabel == target:
                accuracyCount += 1
        testAccuracy = accuracyCount/float(testCount)
        return testAccuracy

    # For binaryMargin is defined as average distance from regression value 
    def margin_distribution(self, X_test, y_test, lambda_index=None, margin_type="regression value"):
        testCount = len(y_test)
        margins_list = []
        misClassifiedVector = np.ones((testCount),dtype=int) # 1 if the example is mis-classified, 0 otherwise 
        accuracyCount = 0
        for i in xrange(testCount):
            point = X_test[i,:]
            target = y_test[i]

            if (margin_type == "regression value"):
                output = self.predict(point, lambda_index)
                margin = abs(output-0.5)
            elif (margin_type == "log ratio"):
                output, one_weight = self.predict(point, lambda_index, get_weight=True)
                zero_weight = Decimal(1) - one_weight
                margin = abs(one_weight.ln()-zero_weight.ln())
            margins_list.append(margin)
            
            if (output > 0.5):
                predLabel = 1
            else:
                predLabel = 0
            if predLabel == y_test[i]:
                accuracyCount += 1
                misClassifiedVector[i] = 0
        testAccuracy = accuracyCount/float(testCount)
        return (testAccuracy, misClassifiedVector, margins_list) 