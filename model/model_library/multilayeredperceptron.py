import math
import random
import numpy
import sys

import scipy




sys.path.append("")
from mathfunction import functions


class SinglePerceptron():
    def __init__(self, input, weight, bias, error):
        self.input = input

        self.weight = weight 
        self.bias = bias

        self.error= error

    def output(self):
        return (self.weight*self.input +self.bias)
    
    def training(self, learning_rate):

        self.weight = random.randfloat(100)
        self.bias = random.randfloat(100)

        gradient = numpy.gradient([self.output(k) for k in self.input])

        while gradient >= self.error:
            diff_weights = learning_rate*self.weights
            self.weights += diff_weights

            diff_bias = learning_rate*self.bias
            self.weights += diff_bias


class MultiLayerPerceptron():

    def __init__(self, layers, input_weights, input_biases, hidden_weights, hidden_biases, output_weights, output_biases):

        self.input_weights = input_weights
        self.input_biases = input_biases

        self.hidden_wheights = hidden_weights
        self.hidden_biases = hidden_biases

        self.output_weights = output_weights
        self.output_biases = output_biases

        self.layers = layers



        
    def neuron(self, input_values, id):
        """single neuron implemented"""

        sum = input_values[id]*self.input_weights[id] + self.input_biases[id]
        new_value = functions.Relu(sum)

        return sum
    
    def sgd(self, loss_function):
        initial_weights = [random.randfloat(100)]*self.layers
        initial_bias = [random.randfloat(100)]*self.layers

        weights = initial_weights
        bias = initial_bias

        #while loss_function()

        
        pass

      


        



