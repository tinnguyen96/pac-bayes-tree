################################################################################
# Author:    Tin Nguyen
# Name:  Node_A.py
# Most recent update: 12/30/2017
#  
# Description: 
################################################################################

import numpy as np

class Node:

    ## Input: rect = Rect object, depth is an int specifying the depth, parent is the parent node
    ## lambda_count is number of lambda parameters to be considered in the validation process 
    def __init__(self, parent, threshold, axis, depth, lambdaCount):
        ## Reference to parent in the template tree
        self._parent = parent
        self._threshold = threshold
        self._axis = axis
        ## The number of splits it took, from the starting unit cube, to get to the node  
        self._depth = depth
        ## Reference to left child
        self._left = None
        ## Reference to right child
        self._right = None
        ## Count of training examples in rectangle by labels 
        self._trainInfo = 0
        self._regGap = 0
        self._valInfo = 0
        ## Label (either 0 or 1) which minimizes empirical error
        self._label = 0
        ## Penalty of the node
        self._penalty = 0
        self._prior = 0
        ## Empirical error
        self._phi = 0
        self._weight = 0
        self._alpha = 0
        self._beta = 0
        ## Whether or not the node has been visited in the bottom up pass of aggregate weighting or the pruning process 
        self._visited = False

    ## the deeper the nodes, the smaller the priority they have in the min-priority queue. This is to ensure
    ## that parent nodes are processed after their children 
    def __lt__(self, other):
        return (other._depth < self._depth)

    def __gt__(self, other):
	return (other._depth > self._depth) 

    def __eq__(self, other):
	return (other._depth == self._depth)       

    def get_axis(self):
        return self._axis

    def get_threshold(self):
        return self._threshold

    def get_depth(self):
        return self._depth

    def get_parent(self):
        return self._parent

    def set_left(self, node):
        self._left = node

    def get_left(self):
        return self._left

    def set_right(self, node):
        self._right = node

    def get_right(self):
        return self._right

    def set_trainInfo(self, trainInfo):
        self._trainInfo = trainInfo

    def get_trainInfo(self):
        return self._trainInfo

    def set_regGap(self, regGap):
        self._regGap = regGap

    def get_regGap(self):
        return self._regGap

    def set_valInfo(self, valInfo):
        self._valInfo = valInfo

    def get_valInfo(self):
        return self._valInfo

    def set_prior(self, prior):
        self._prior = prior

    def get_prior(self):
        return self._prior

    def set_label(self, label):
        self._label = label

    def get_label(self):
        return self._label

    def set_empError(self, empErr):
        self._empErr = empErr

    def get_empError(self):
        return self._empErr

    def set_penalty(self, penalty):
        self._penalty = penalty

    def get_penalty(self):
        return self._penalty

    def set_phi(self, phi):
        self._phi = phi

    def get_phi(self):
        return self._phi

    def set_weight(self, weight):
        self._weight = weight

    def get_weight(self):
        return self._weight

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_alpha(self):
        return self._alpha

    def set_beta(self, beta):
        self._beta = beta

    def get_beta(self):
        return self._beta

    def set_visited(self, visited):
        self._visited = visited

    def get_visited(self):
        return self._visited
