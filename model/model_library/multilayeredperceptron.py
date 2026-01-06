import math
import random
import numpy
import sys

import scipy




sys.path.append("")
from mathfunction import functions


class SinglePerceptron():
    def __init__(self, weight, bias, file_directory):
        """
        
        """

        self.weight = weight 
        self.bias = bias

        

    def output(self, input):
        return (self.weight*input +self.bias)
    
    def training(self, learning_rate, number_epoch, cost_function, training_data, batch_size):
        """
        learning_rate: int
        number_epoch: int
        cost_function = function from mathfunctions
        training_data: int 2-tuple list 
        batch_size = int
        """

        self.weight = random.randfloat(100)
        self.bias = random.randfloat(100)

        for k in range(number_epoch):

            shuffled_batch = random.shuffle(training_data)[::batch_size]

            shuffled_batch_applied = [(k[0], self.output(k[0])) for k in shuffled_batch]

            delta_list = [cost_function(k[0], k[1]) for k in shuffled_batch_applied]

            grad = numpy.gradient(delta_list)

            self.weight -= grad
            self.bias -= grad

        return self.weight, self.bias

        

class MultiLayerPerceptron(SinglePerceptron):

    def __init__(self, layers):
        """
        layers = int list
        """

        
        self.layers = layers
        
    

        

      


        



