################################################################################
# Author: Tin Nguyen, Samory Kpotufe. 
# Name:  dyadic_pacbayes.py 
#
# Feature: 
# 
# Usage guideline:
# -
# Future work:
# - instead of using Decimal class to store the messages alpha and beta, does it 
# make sense to normalize them along the dynamic program?  
# 
# Dependencies: Node.py
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

class DDT:
    ## Input: 
    def __init__(self, numOfLabels, penaltyType="ada", priorType="effInt", val_ratio=0.35, 
                lambda_range=np.append(np.logspace(-8, 6, num=10, base=2.0),1), error_ratio=0.5, 
                fullSample=True, linear_search=True, check_invariants=False, report_lambda=True):
        self._leaves, self._alter_leaves = Queue.PriorityQueue(), Queue.PriorityQueue()
        self._base_lambda_range, self._lambda_count = lambda_range, lambda_range.size
        self._error_ratio  = error_ratio
        self._numOfLabels = numOfLabels
        self._penaltyType = penaltyType
        self._priorType = priorType
        self._alterType = alterType
        self._linear_search = linear_search
        self._fullSample = fullSample
        self._check_invariants = check_invariants 
        self._report_lambda = report_lambda
        self._val_ratio = val_ratio

    ## Input: X = data component of sample, y = label component
    def fit(self, X, y, random_state=42):
        # First split X into a train and validation component to determine best lambda paramaters 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self._val_ratio, random_state=random_state)
            
        # Instantiate instance variables 
        self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)
        self._X_val, self._y_val, self._valSize = X_val, y_val, len(y_val)
        self._dim = X_train.shape[1]
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)
        self._unitCube = np.zeros((self._dim,2),dtype=np.int)

        # alternating optimization to search for temperature parameters 
        self.__logSearch_errLambda(True)
        self.__linearSearch_errLambda()
        self.__logSearch_penLambda(False)
        self.__linearSearch_penLambda()

        # Debugging
        if (self._check_invariants):
            for lambdaIdx in xrange(self._lambda_count):
                print "Traverse the template tree for the %s lambda value" %str(lambdaIdx)
                self.__traverse(self._root, 0, 5, lambdaIdx)
            
        # Debugging
        if (self._check_invariants):
            for lambdaIdx in xrange(self._lambda_count):
                print "Check the correctness of the dynamic program for the %s lambda value" %str(lambdaIdx)
                print self.checkWeights(self._root, lambdaIdx)
        
        # After the best lambda parameters have been found, train the aggregate on the whole sample 
        if (self._fullSample):
            X_train = np.concatenate((X_train,X_val),axis=0)
            y_train = np.append(y_train,y_val)
            self._X_train, self._y_train, self._trainSize = X_train, y_train, len(y_train)
            self.__fullSample()

    def __logSearch_penLambda(self, isFirst):
        # construct template tree
        self._pen_lambda_range = np.copy(self._base_lambda_range)
        if (isFirst):
            self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)
            self._err_lambda_range = 1
        else:
            if (self._alterType == "mixedErrFirst"):
                self._err_lambda_range = self._best_err_lambda
            # regain access to terminal leaves of template tree 
            self._leaves = self._alter_leaves
            self._alter_leaves = Queue.PriorityQueue()
            self._err_lambda_range = self._best_err_lambda

        # two-pass dynamic program
        self.__bottom_up_pass()
        self.__top_down_pass()

        # Debugging
        if (self._check_invariants):
            for lambdaIdx in xrange(self._lambda_count):
                print "Check the correctness of the dynamic program for the %s lambda value" %str(lambdaIdx)
                print self.checkWeights(self._root, lambdaIdx)

        # optimize lambda
        self._best_model_index, self._best_val_err, self._val_err = self.__optimize_lambda()
        self._best_pen_lambda = self._pen_lambda_range[self._best_model_index]

        if (self._report_lambda):
            print "Lambda range after log search for penalty-lambda %s" %str(self._pen_lambda_range)
            print "Validation errors after log search for penalty-lambda %s" %str(self._val_err)
            print "Value of best penalty-lambda %s" %str(self._best_pen_lambda)

    def __linearSearch_penLambda(self):
        if (self._linear_search):
            self._pen_lambda_range = np.linspace(self._best_pen_lambda/2,self._best_pen_lambda*2, num=self._lambda_count)

            # regain access to terminal leaves of template tree
            self._leaves = self._alter_leaves
            self._alter_leaves = Queue.PriorityQueue()

            # two-pass dynamic program
            self.__bottom_up_pass()
            self.__top_down_pass()

            # Debugging
            if (self._check_invariants):
                for lambdaIdx in xrange(self._lambda_count):
                    print "Check the correctness of the dynamic program for the %s lambda value" %str(lambdaIdx)
                    print self.checkWeights(self._root, lambdaIdx)

            # optimize lambda 
            self._best_model_index, self._best_val_err, self._val_err = self.__optimize_lambda()

            # check that the error around the lambda value doesn't change too much 
            if (self._best_model_index != self._lambda_count - 1):
                curr = self._best_model_index
                while ((self._val_err[curr+1]-self._val_err[curr])/self._val_err[curr] > self._error_ratio):
                    curr = curr - 1
                self._best_model_index = curr
            self._best_pen_lambda = self._pen_lambda_range[self._best_model_index]

            if (self._report_lambda):
                print "Lambda range after linear search for penalty-lambda %s" %str(self._pen_lambda_range)
                print "Validation errors after linear search for penalty-lambda %s" %str(self._val_err)
                print "Value of best penalty-lambda %s" %str(self._best_pen_lambda)

    def __logSearch_errLambda(self, isFirst):
        self._err_lambda_range = np.copy(self._base_lambda_range)
        if (isFirst):
            self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)
            self._pen_lambda_range = 1
        else:
            if (self._alterType == "mixedPenFirst"):
                self._pen_lambda_range = self._best_pen_lambda
            # regain access to terminal leaves of template tree 
            self._leaves = self._alter_leaves
            self._alter_leaves = Queue.PriorityQueue()
            self._pen_lambda_range = self._best_pen_lambda

        # two-pass dynamic program
        self.__bottom_up_pass()
        self.__top_down_pass()
        
        # Debugging
        if (self._check_invariants):
            for lambdaIdx in xrange(self._lambda_count):
                print "Check the correctness of the dynamic program for the %s lambda value" %str(lambdaIdx)
                print self.checkWeights(self._root, lambdaIdx)

        # optimize lambda 
        self._best_model_index, self._best_val_err, self._val_err = self.__optimize_lambda()
        self._best_err_lambda = self._err_lambda_range[self._best_model_index]

        if (self._report_lambda):
            print "Lambda range after log search for error-lambda %s" %str(self._err_lambda_range)
            print "Validation errors after log search for error-lambda %s" %str(self._val_err)
            print "Value of best error-lambda %s" %str(self._best_err_lambda)

    def __linearSearch_errLambda(self):
        if (self._linear_search):
            self._err_lambda_range = np.linspace(self._best_err_lambda/2,self._best_err_lambda*2, num=self._lambda_count)

            # regain access to terminal leaves of template tree
            self._leaves = self._alter_leaves
            self._alter_leaves = Queue.PriorityQueue()

            # two-pass dynamic program
            self.__bottom_up_pass()
            self.__top_down_pass()

            # Debugging
            if (self._check_invariants):
                for lambdaIdx in xrange(self._lambda_count):
                    print "Check the correctness of the dynamic program for the %s lambda value" %str(lambdaIdx)
                    print self.checkWeights(self._root, lambdaIdx)

            # optimize lambda 
            self._best_model_index, self._best_val_err, self._val_err = self.__optimize_lambda()

            # check that the error around the lambda value doesn't change too much 
            if (self._best_model_index != self._lambda_count - 1):
                curr = self._best_model_index
                while ((self._val_err[curr+1]-self._val_err[curr])/self._val_err[curr] > self._error_ratio) and (curr > 0):
                    curr = curr - 1
                self._best_model_index = curr
            self._best_err_lambda = self._err_lambda_range[self._best_model_index]

            if (self._report_lambda):
                print "Lambda range after linear search for error-lambda %s" %str(self._err_lambda_range)
                print "Validation errors after linear search for error-lambda %s" %str(self._val_err)
                print "Value of best error-lambda %s" %str(self._best_err_lambda)

    def __fullSample(self):
        self._pen_lambda_range, self._err_lambda_range = np.array([self._best_pen_lambda]), self._best_err_lambda
        self._lambda_count = 1
        self._best_model_index = 0
        self._leaves = Queue.PriorityQueue()
        
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)

        # construct template tree
        self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)

        # two-pass dynamic program
        self.__bottom_up_pass()
        self.__top_down_pass()

    def __lambdaSensitivity(self):
        self._pen_lambda_range, self._err_lambda_range = np.array([self._best_pen_lambda]), self._best_err_lambda
        self._lambda_count = 1
        self._leaves = Queue.PriorityQueue()
        
        self._max_splits = int(np.floor(np.log(self._trainSize)))
        self._treeArr = range(self._trainSize)

        # construct template tree
        self._root = self.__grow(None, self._unitCube, 0, 0, 0, self._trainSize)

        # two-pass dynamic program
        self.__bottom_up_pass()
        self.__top_down_pass()

    ## method to update empirical error  
    def __update_empError_regGap(self, node):
        trainInfo = node.get_trainInfo()
        label = np.argmax(trainInfo)
        node.set_label(label)
        empErr = (np.sum(trainInfo)-trainInfo[label])/float(self._trainSize)
        node.set_empError(empErr)

        exCount = np.sum(trainInfo)
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
            probMass = exCount/float(self._trainSize)
            codeLength = node.get_depth()
            temp = (codeLength)/float(self._trainSize)
            pHat = 4*np.max([probMass, temp])
            penalty = np.sqrt(2*pHat*temp)
        else:
            penalty = 0
        node.set_penalty(penalty)

    ## method to update prior for all nodes
    def __update_prior(self, node):
        exCount = np.sum(node.get_trainInfo())
        if (exCount > 0):
            prior = np.exp(-1)
        else:
            prior = 1 
        node.set_prior(prior)

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
        self.__update_empError_regGap(currNode)
        self.__update_penalty(currNode)
        self.__update_prior(currNode)
        return currNode

    ## bottom up pass of aggregate weighting: update phi and beta 
    def __bottom_up_pass(self):
        while not self._leaves.empty():
            node = self._leaves.get() ## because self._leaves is PQ by node depth, will always process both siblings before their parent 
            node.set_visited(False)

            ## corner case: n is a terminal node of the template tree
            if node.get_threshold() is None:
                self._alter_leaves.put(node)
                temp = map(Decimal, np.zeros(self._lambda_count))
            else:
                left = node.get_left()
                right = node.get_right()
                temp = []
                for lambdaIdx in xrange(self._lambda_count):
                    betaLeft = left.get_beta()[lambdaIdx]
                    betaRight = right.get_beta()[lambdaIdx]
                    temp.append(betaLeft*betaRight)

            expPhi = np.exp(-(node.get_penalty()*self._pen_lambda_range+node.get_empError()*self._err_lambda_range)*self._trainSize)*node.get_prior()
            expPhi = map(Decimal, expPhi)
            # print "expPhi value is %s" %str(expPhi)
            beta = []
            for lambdaIdx in xrange(self._lambda_count):
                beta.append(temp[lambdaIdx]+expPhi[lambdaIdx])
            node.set_phi(expPhi)
            node.set_beta(beta)
            # print "Beta value is %s" %str(beta)
            parent = node.get_parent()
            ## corner case: reaching the root node 
            if (parent is None):
                return
            else:
                if not parent.get_visited(): 
                    parent.set_visited(True)
                    self._leaves.put(parent)

    ## top down pass of aggregate weighting: update alpha and weight 
    def __top_down_pass(self):
        levelOrder = Queue.Queue()
        rootAlpha = map(Decimal, np.ones(self._lambda_count))
        expPhiRoot = self._root.get_phi()
        weightsRoot = []
        for lambdaIdx in xrange(self._lambda_count):
            weightsRoot.append(expPhiRoot[lambdaIdx]*rootAlpha[lambdaIdx])
        self._root.set_alpha(rootAlpha)
        # print "Root alpha %s" %str(rootAlpha)
        self._root.set_weight(weightsRoot)
        # print "Root weights %s" %str(weightsRoot)
        levelOrder.put(self._root)
        while not levelOrder.empty():
            n = levelOrder.get() ## because levelOrder is FIFO queue, will do level order traversal of the template tree 

            if not (n.get_threshold() is None):
                left = n.get_left()
                right = n.get_right()

                expPhiLeft = left.get_phi()
                # print "expPhiLeft are %s" %str(expPhiLeft)
                alphaLeft = []
                weightsLeft = []
                for lambdaIdx in xrange(self._lambda_count):
                    parentAlpha = n.get_alpha()[lambdaIdx]
                    betaRight = right.get_beta()[lambdaIdx]
                    temp = parentAlpha*betaRight
                    alphaLeft.append(temp) 
                    weightsLeft.append(expPhiLeft[lambdaIdx]*temp)
                left.set_alpha(alphaLeft)
                # print "Left alpha are %s" %str(alphaLeft)
                left.set_weight(weightsLeft)
                # print "Left weights are %s" %str(weightsLeft)

                expPhiRight = right.get_phi()
                # print "expPhiRight are %s" %str(expPhiRight)
                alphaRight = []
                weightsRight = []
                for lambdaIdx in xrange(self._lambda_count):
                    parentAlpha = n.get_alpha()[lambdaIdx]
                    betaLeft = left.get_beta()[lambdaIdx]
                    temp = parentAlpha*betaLeft
                    alphaRight.append(temp) 
                    weightsRight.append(expPhiRight[lambdaIdx]*temp)
                right.set_alpha(alphaRight)
                # print "Right alpha are %s" %str(alphaRight)
                right.set_weight(weightsRight)
                # print "Right weights are %s" %str(weightsRight)

                levelOrder.put(left)
                levelOrder.put(right)

    def get_path(self, point, lambda_index = None):
        if lambda_index is None:
            model_index = self._best_model_index
        else:
            model_index = lambda_index
        train_info_list = []
        exp_phi_list = []
        weight_list = []
        alpha_list = []
        beta_list = []

        currNode = self._root
        labelWeights = map(Decimal, np.zeros(self._numOfLabels))

        ## follow the path starting at root to get to the leaf that contains the point'
        while (True):
            label = currNode.get_label()
            labelWeights[label] = labelWeights[label] + currNode.get_weight()[model_index]
            train_info_list.append(currNode.get_trainInfo())
            exp_phi_list.append(currNode.get_phi()[model_index])
            weight_list.append(currNode.get_weight()[model_index])
            beta_list.append(currNode.get_beta()[model_index])
            alpha_list.append(currNode.get_alpha()[model_index])

            threshold = currNode.get_threshold()
            if (threshold is None):
                break
            else:
                axis = currNode.get_axis()
                if (point[axis] > threshold):
                    currNode = currNode.get_right()
                else:
                    currNode = currNode.get_left()
        return (labelWeights, exp_phi_list, train_info_list, weight_list, alpha_list, beta_list)

    def __class_weights_and_average_gap(self, point, lambda_index=None):
        if lambda_index is None:
            model_index = self._best_model_index
        else:
            model_index = lambda_index

        currNode = self._root
        labelWeights = map(Decimal, np.zeros(self._numOfLabels))
        gapSum = map(Decimal, np.zeros(self._numOfLabels))

        ## follow the path starting at root to get to the leaf that contains the point'
        while (True):
            ## don't consider contributions of empty cells 
            if (np.sum(currNode.get_trainInfo()) == 0):
                break
            else:
                label = currNode.get_label()
                labelWeights[label] += currNode.get_weight()[model_index]
                gapSum[label] += Decimal(currNode.get_regGap())*currNode.get_weight()[model_index]
                threshold = currNode.get_threshold()
                if (threshold is None):
                    break
                else:
                    axis = currNode.get_axis()
                    if (point[axis] > threshold):
                        currNode = currNode.get_right()
                    else:
                        currNode = currNode.get_left()

        return (labelWeights, gapSum)

    ## Output: use the validation set to determine the best lambda parameter and the correspondinng 
    ## validation error of the aggregate 
    def __optimize_lambda(self):
        # print lambda_count
        valErrCount = np.zeros((self._lambda_count))
        for j in xrange(self._valSize):
            point = self._X_val[j]
            target = self._y_val[j]
            
            for lambdaIndex in xrange(self._lambda_count):
                labelWeights, _ = self.__class_weights_and_average_gap(point,lambdaIndex)
                predLabel = labelWeights.index(max(labelWeights))
                if target != predLabel:
                    valErrCount[lambdaIndex] += 1

        valErr = valErrCount/float(self._valSize)
        best_model_index = np.argmin(valErr)
        best_val_err = valErr[best_model_index]
        return (best_model_index, best_val_err, valErr)

    def __traverse(self, node, minDepth, maxDepth, lambdaIdx):
        if (minDepth <= node.get_depth()) and (node.get_depth() <= maxDepth):
            print "axis %s" %str(node.get_axis())
            print "threshold %s" %str(node.get_threshold())
            print "empErr %s" %str(node.get_empError())
            print "penalty %s" %str(node.get_penalty())
            print "expPhi %s" %str(node.get_phi()[lambdaIdx])
            print "beta %s" %str(node.get_beta()[lambdaIdx])
            print "alpha %s" %str(node.get_alpha()[lambdaIdx])
            print "weights %s" %str(node.get_weight()[lambdaIdx])
            if node.get_threshold() is None:
                print "Reached terminal leaf of depth %s" %str(node.get_depth())
                return 
            else:
                self.__traverse(node.get_left(), minDepth, maxDepth, lambdaIdx)
                self.__traverse(node.get_right(), minDepth, maxDepth, lambdaIdx)
        else:
            return 
            
    def checkWeights(self, node, lambdaIdx, tol=1e-10):
        betaParent = node.get_beta()[lambdaIdx]
        phiParent = node.get_phi()[lambdaIdx]
        alphaParent = node.get_alpha()[lambdaIdx]
        weightsParent = node.get_weight()[lambdaIdx]
        weightsCorrect = (abs(weightsParent-phiParent*alphaParent) < tol)
        if (node.get_threshold() is None):
            betaCorrect = (abs(phiParent-betaParent) < tol)
            return (betaCorrect and weightsCorrect)
        else:
            left = node.get_left()
            right = node.get_right()
            
            betaLeft = left.get_beta()[lambdaIdx]
            betaRight = right.get_beta()[lambdaIdx]
            
            phiLeft = left.get_phi()[lambdaIdx]
            phiRight = right.get_phi()[lambdaIdx]

            alphaLeft = left.get_alpha()[lambdaIdx]
            alphaRight = right.get_alpha()[lambdaIdx]

            weightsLeft = left.get_weight()[lambdaIdx]
            weightsRight = right.get_weight()[lambdaIdx]

            betaCorrect = (abs(betaLeft*betaRight+phiParent-betaParent) < tol)
            alphaCorrect = (abs(alphaLeft-alphaParent*betaRight) < tol) and (abs(alphaRight-alphaParent*betaLeft) < tol)
            alphaCorrect = True
            if not (alphaCorrect and betaCorrect and weightsCorrect):
                return False
                print node.get_depth()
            else:
                return (self.checkWeights(left,lambdaIdx) and self.checkWeights(right,lambdaIdx))     


    ## Input: X_test = data component of test set, y_test = label component
    ## Output: test accuracy using the best aggregate classifier 
    def proba_weights(self, X_test, y_test, lambda_index=None):
        testCount = len(y_test)
        weights_list = []
        accuracyCount = 0
        for i in xrange(testCount):
            labelWeights, _ = self.__class_weights_and_average_gap(X_test[i,:],lambda_index)
            weights_list.append(labelWeights)
        return weights_list

    def margin_distribution(self, X_test, y_test, margin_type="log ratio", lambda_index=None):
        testCount = len(y_test)
        margins_list = []
        misClassifiedVector = np.ones((testCount),dtype=int) # 1 if the example is mis-classified, 0 otherwise 
        accuracyCount = 0
        for i in xrange(testCount):
            labelWeights, gapSum = self.__class_weights_and_average_gap(X_test[i,:],lambda_index)

            logWeights = []
            for j in xrange(self._numOfLabels):
                logWeights.append(labelWeights[j].ln())
            predLabel = logWeights.index(max(logWeights))

            if (margin_type == "log ratio"):
                highlestLogWeight = logWeights[predLabel]
                logWeights.pop(predLabel)
                largestLogWeightOtherwise = max(logWeights)
                if (highlestLogWeight == Decimal('-Infinity')):
                    margin = 0 
                else:
                    margin = highlestLogWeight - largestLogWeightOtherwise
            elif (margin_type == "average gap"):
                margin = gapSum[predLabel]/labelWeights[predLabel]

            # update margin list 
            margins_list.append(margin)

            if predLabel == y_test[i]:
                accuracyCount += 1
                misClassifiedVector[i] = 0
        testAccuracy = accuracyCount/float(testCount)
        return (testAccuracy, misClassifiedVector, margins_list)

    ## Input: X_test = data component of test set, y_test = label component
    ## Output: test accuracy using the best aggregate classifier 
    def score(self, X_test, y_test, lambda_index=None):
        testCount = len(y_test)
        accuracyCount = 0
        for i in xrange(testCount):
            labelWeights, _ = self.__class_weights_and_average_gap(X_test[i,:],lambda_index)
            predLabel = labelWeights.index(max(labelWeights))
            if predLabel == y_test[i]:
                accuracyCount += 1
        testAccuracy = accuracyCount/float(testCount)
        return testAccuracy