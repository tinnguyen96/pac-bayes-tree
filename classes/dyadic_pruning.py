################################################################################
# Author: Tin Nguyen, Samory Kpotufe. 
# Name: dyadic-pruning.py 
#
# Feature: 
# - classify using deepest node which contains data 
# 
# Usage guideline:
# -
# Dependencies: Node_P.py 
################################################################################
import numpy as np
import Queue 
import time
import Node_P as Node
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

class DDT:
    ## Input: 
    def __init__(self, numOfLabels, penaltyType, val_ratio=0.35, lambda_range=np.logspace(-8, 6, num=10, base=2.0), 
        linear_search=True):
        self._leaves, self._alter_leaves = Queue.PriorityQueue(), Queue.PriorityQueue()
        self._lambda_range, self._lambda_count = lambda_range, lambda_range.size
        self._numOfLabels = numOfLabels
        self._penaltyType = penaltyType
        self._linear_search = linear_search
        self._val_ratio = val_ratio

    ## Input: X = data component of sample, y = label component
    def fit(self, X, y, random_state=42):
        ## log-search for lambda: split sample into training and validating parts to select best scale param
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self._val_ratio, random_state=random_state)
            
        # set instance variables 
        self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)
        self._X_val, self._y_val, self._valSize = X_val, y_val, len(y_val)
        self._dim = X_train.shape[1]
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)

        # grow template tree 
        unitCube = np.zeros((self._dim,2),dtype=np.int)
        start = time.time()
        self._root = self.__grow(None, unitCube, 0, 0, 0, self._trainSize)
        growtime = time.time() - start
        # print "log-search: Finished clever growing, time elapsed = %s" %str(growtime)

        # prune template tree 
        # print "log-search: Start pruning"
        start = time.time()
        self.__prune()
        prunetime = time.time() - start
        # print "log-search: Finished pruning, time elapsed = %s" %str(growtime)

        # Debugging
        if (True):
            for lambdaIdx in xrange(self._lambda_count):
                # print "Traverse the pruning for the %s lambda value" %str(lambdaIdx)
                # self.__traverse(self._root, lambdaIdx)
                print "Check that optimal error and penalty decomposes properly for the %s lambda value" %str(lambdaIdx)
                print self.checkOptimalDecomposition(self._root, lambdaIdx)

        # print "log-search: Use validation set to select best scale parameter"
        start = time.time()
        self._best_model_index, self._best_lambda, self._best_val_err, self._val_err = self.__optimize_lambda()
        # print "log-search: Log-search returns lambda = %s" %str(self._best_lambda)
        optimizetime = time.time() - start

        if (self._linear_search):
            self._lambda_range = np.linspace(self._best_lambda/2,self._best_lambda*2, num=self._lambda_count)
            # print "linear-search: Start pruning"
            self._leaves = self._alter_leaves
            self._alter_leaves = Queue.PriorityQueue()

            start = time.time()
            self.__prune()
            prunetime = time.time() - start
            # print "linear-search: Finished pruning, time elapsed = %s" %str(growtime)

            # print "linear-search: Use validation set to select best scale parameter"
            start = time.time()
            self._best_model_index, self._best_lambda, self._best_val_err, self._val_err = self.__optimize_lambda()
            # print "linear-search: Linear-search returns lambda = %s" %str(self._best_lambda)
            optimizetime = time.time() - start

        if (True):
            print "Validation errors are %s" %str(self._val_err)
            print "Lambda range after grid search %s" %str(self._lambda_range)

        ## Use whole sample to train, now knowing best scale param
        X_train = np.concatenate((X_train,X_val),axis=0)
        y_train = np.append(y_train,y_val)
        
        self._lambda_range, self._lambda_count = np.array([self._best_lambda]), 1
        self._best_model_index = 0
        self._leaves = Queue.PriorityQueue()
        self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)

        # print "Full-sample: Start growing"
        # print unitCube
        start = time.time()
        self._root = self.__grow(None, unitCube, 0, 0, 0, self._trainSize)
        growtime = time.time() - start
        # print "Full-sample: Finished clever growing, time elapsed = %s" %str(growtime)

        # print "Full-sample: Start pruning"
        start = time.time()
        self.__prune()
        prunetime = time.time() - start
        # print "Full-sample: Finished pruning, time elapsed = %s" %str(growtime)

        if (True):
            trainError = 1 - self.score(X_train,y_train)
            print "Training error is %s" %str(trainError)
            start = time.time()
            self._best_model_index, self._best_lambda, self._best_val_err, self._val_err = self.__optimize_lambda()
            # print "linear-search: Linear-search returns lambda = %s" %str(self._best_lambda)
            optimizetime = time.time() - start
            print "Validation errors are %s" %str(self._val_err)
            print "Lambda range after grid search %s" %str(self._lambda_range)

    ## method to update empirical error  
    def __update_empError_regGap(self, node):
        trainInfo = node.get_trainInfo()
        label = np.argmax(trainInfo)
        exCount = np.sum(trainInfo)
        node.set_label(label)
        empErr = (exCount-trainInfo[label])/float(self._trainSize)
        node.set_empError(empErr)
        if (exCount > 0):
            sortedTrainInfo = np.sort(trainInfo) # two largest label counts are at the end of the array 
            gap = (sortedTrainInfo[self._numOfLabels-1]-sortedTrainInfo[self._numOfLabels-2])/float(exCount)
        else:
            gap = 0
        node.set_regGap(gap)    
        
    ## method to update penalty 
    def __update_penalty(self, node):
        exCount = np.sum(node.get_trainInfo())
        if (exCount > 0):
            if (self._penaltyType == "heuristic"):
                penalty = 1/float(self._trainSize)
            elif (self._penaltyType == "SN"):
                probMass = exCount/float(self._trainSize)
                codeLength = node.get_depth()
                temp = (codeLength)/float(self._trainSize)
                pHat = 4*np.max([probMass, temp])
                penalty = np.sqrt(2*pHat*temp)
        else:
            penalty = 0
        node.set_penalty(penalty)

    def __initialize_optimal(self, node):
        for lambdaIdx in xrange(self._lambda_count):
            node.set_optError(node.get_empError(),lambdaIdx)
            node.set_optPen(node.get_penalty(),lambdaIdx)
            node.set_isOptimal(True, lambdaIdx)

    def __should_merge(self, node):
        left = node.get_left()
        right = node.get_right()
        # print node.get_depth()
        for lambdaIdx in xrange(self._lambda_count):
            parentObj = node.get_empError() + self._lambda_range[lambdaIdx]*node.get_penalty()
            childrenErr = left.get_optError(lambdaIdx) + right.get_optError(lambdaIdx)
            childrenPen = left.get_optPen(lambdaIdx) + right.get_optPen(lambdaIdx)
            if (False):
                print "Objective at node %s" %str(parentObj)
                print "Sum of errs of children %s" %str(childrenErr)
                print "Optimal error of left child %s" %str(left.get_optError(lambdaIdx))
                print "Optimal error of right child %s" %str(right.get_optError(lambdaIdx))
                print "Sum of penalties of children %s" %str(childrenPen)
                print "Optimal penalty of left child %s" %str(left.get_optPen(lambdaIdx))
                print "Optimal penalty of right child %s" %str(right.get_optPen(lambdaIdx))
            if (parentObj <= childrenErr + self._lambda_range[lambdaIdx]*childrenPen):
                node.set_optError(node.get_empError(),lambdaIdx)
                node.set_optPen(node.get_penalty(),lambdaIdx)
                node.set_isOptimal(True,lambdaIdx)
            else:
                node.set_optError(childrenErr,lambdaIdx)
                node.set_optPen(childrenPen,lambdaIdx)              
                node.set_isOptimal(False,lambdaIdx)

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
                if (self._X_train[exampleIdx,axis] >= threshold):
                    leftHi, rightLo = splitIdx, splitIdx
                    # print exampleIdx
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
        self.__update_empError_regGap(currNode)
        self.__update_penalty(currNode)
        self.__initialize_optimal(currNode)
        # print lo, hi
        # print currNode.get_depth()
        # print currNode.get_trainInfo()
        return currNode

    ## prune the template tree 
    def __prune(self):
        while not self._leaves.empty():
            node = self._leaves.get() ## because self._leaves is PQ by node depth, will always process both siblings before their parent 
            # print "Popped from the queue node of depth %s" %str(node.get_depth())
            node.set_visited(False)

            ## corner case: n is a terminal node of the template tree
            if node.get_threshold() is None:
                self._alter_leaves.put(node)

            parent = node.get_parent()
            if (not (parent is None)):
                if not parent.get_visited(): 
                    parent.set_visited(True)
                    # print "Checked should merge on parent of depth %s" %str(parent.get_depth())
                    self.__should_merge(parent)
                    self._leaves.put(parent) 

    def __which_cell(self, point, lambdaIndex):
        prevNode = None 
        currNode = self._root
        if lambdaIndex is None:
            modelIndex = self._best_model_index
        else:
            modelIndex = lambdaIndex
        ## follow the path starting at root to get to the leaf that contains the point
        while not currNode.get_isOptimal(modelIndex):
            prevNode = currNode
            threshold = currNode.get_threshold()
            # print threshold
            axis = currNode.get_axis()
            if (point[axis] > threshold):
                currNode = currNode.get_right()
            else:
                currNode = currNode.get_left()
            # print currNode.get_depth()
        # don't use empty leaf nodes for classification 
        if (np.sum(currNode.get_trainInfo()) == 0):
            currNode = prevNode 
        return currNode

    ## select the pruning that minimizes error on validation set 
    def __optimize_lambda(self):
        valErrCount = np.zeros((self._lambda_count))
        for lambdaIndex in xrange(self._lambda_count):
            for exampleIdx in xrange(self._valSize):
                node = self.__which_cell(self._X_val[exampleIdx],lambdaIndex)
                predLabel = node.get_label()
                if predLabel != self._y_val[exampleIdx]:
                    valErrCount[lambdaIndex] += 1
        valErr = valErrCount/float(self._valSize)
        best_model_index = np.argmin(valErr, axis=0)
        best_val_err = np.amin(valErr, axis=0)
        best_lambda = self._lambda_range[best_model_index]
        return (best_model_index, best_lambda, best_val_err, valErr)

    def __traverse(self, node, lambdaIndex):
        print "depth %s" %str(node.get_depth())
        # print "axis %s" %str(node.get_axis())
        # print "threshold %s" %str(node.get_threshold())
        print "empErr %s" %str(node.get_empError())
        print "optErr %s" %str(node.get_optError(lambdaIndex))
        print "penalty %s" %str(node.get_penalty())
        print "optPen %s" %str(node.get_optPen(lambdaIndex))
        print "trainInfo %s" %str(node.get_trainInfo())
        print "This node is optimal: %s" %str(node.get_isOptimal(lambdaIndex))
        if node.get_threshold() is None:
            print "Reached terminal leaf of depth"
            return 
        else:
            self.__traverse(node.get_left(),lambdaIndex)
            self.__traverse(node.get_right(),lambdaIndex)

    def checkOptimalDecomposition(self, node, lambdaIndex, tol=1e-6):
        if (node.get_threshold() is None):
            return True
        left = node.get_left()
        right = node.get_right()
        if not node.get_isOptimal(lambdaIndex):
            errDiff = abs(node.get_optError(lambdaIndex) -  left.get_optError(lambdaIndex) - right.get_optError(lambdaIndex))
            penDiff = abs(node.get_optPen(lambdaIndex) -  left.get_optPen(lambdaIndex) - right.get_optPen(lambdaIndex))
        else:
            errDiff = abs(node.get_optError(lambdaIndex) - node.get_empError())
            penDiff = abs(node.get_optPen(lambdaIndex) - node.get_penalty())
        correctness = (errDiff < tol) and (penDiff < tol)
        if not correctness:
            return False
        else:
            return (self.checkOptimalDecomposition(left, lambdaIndex) and self.checkOptimalDecomposition(right, lambdaIndex))

    def margin_distribution_for_abstain(self, X_test, y_test, lambda_index=None):
        testCount = len(y_test)
        margins_list = []
        accuracyCount = 0
        misClassifiedVector = np.ones((testCount),dtype=int)
        for exampleIdx in xrange(testCount):
            node = self.__which_cell(X_test[exampleIdx],lambda_index)
            regGap = node.get_regGap()
            margins_list.append(regGap) 
            predLabel = node.get_label()
            if predLabel == y_test[exampleIdx]:
                misClassifiedVector[exampleIdx] = 0
                accuracyCount += 1
        testAccuracy = accuracyCount/float(testCount)
        return (testAccuracy, misClassifiedVector, margins_list)

    ## Input: X_test = data component of test set, y_test = label component
    ## Output: test accuracy using the best classifier 
    def score(self, X_test, y_test, lambda_index=None):
        testCount = len(y_test)
        accuracyCount = 0
        for exampleIdx in xrange(testCount):
            node = self.__which_cell(X_test[exampleIdx],lambda_index)
            predLabel = node.get_label()
            if predLabel == y_test[exampleIdx]:
                accuracyCount += 1
        testAccuracy = accuracyCount/float(testCount)
        return testAccuracy