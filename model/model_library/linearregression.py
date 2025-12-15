import math
import os
import sys
import numpy
sys.path.append("")

from mathfunction import functions

class singleperceptron():
    def __init__(self, input, weights, bias):
        """
        input: set of two real numbers
        wheights = real number
        bias: real number
        """

        self.input = input
        self.weights = weights 
        self.bias = bias

    def train(self, loss_function, pas):
        """
        pas = real number
        loss_function = function from mathfunction.py
        """
        data = self.input
        ecart = pas
        
    def outupt(self):
        return self.input*self.weights + self.bias

