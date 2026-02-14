import math
import random
import numpy
import sys
import regex
import os
import scipy


import activation_functions



#sys.path.append("/Users/dossierantoine/Documents/GitHub/dl/model/mathfunction.py")
sys.path.insert(0, os.path.abspath('model'))


class FeedForwardNeuralNetwork():

    def __init__(self, list_depth_layer, inner_layers, hidden_layers_depth, outter_layers, activation_function):
        """
        list_depth_layer : int list -> list containing the depth of each layer

        inner_layers = int 

        number_hidden_layers = int

        hidden_layers =int list 2-uplet list list -> it is a list containing, for each layer, a list containg a 2-uplet of int lists respectively for weights and biases of each neuron

        outter_layers: int
        """
        self.list_depth_layer = list_depth_layer
        
        self.inner_layers = inner_layers

        self.hidden_layers_depth = hidden_layers_depth

        self.outter_layers = outter_layers

        self.activation_function = activation_function


#take all the outputs of a layer and combine them into a column vector
    def assemble(self, id_layer, input_data, list_weights, list_bias):
        total_vector = []
        for k in range(len(self.list_depth_layer[id_layer])):
            total_vector.append(self.feedforward_singlelayer(id_layer, input_data, list_weights, list_bias))
        return total_vector



    def feedforward_singlelayer(self, id_layer, input_data, list_weights, list_bias):
        #list_weights, list_bias are temporary structures
        """
        nb_layer : the number of this layer in the list of the hidden layers
        input_data : int vect (it is a column vector whose size is equal to hidden_layers_depth)
        """
        #creating a column vector of size hidden_layers_depth
        output = [0 for k in range(self.list_depth_layer[id_layer])]
        result = 0
    
        for k in range(self.list_depth_layer[id_layer]):
            result += list_weights[k]*input_data[k] + list_bias[k]

        return result
        
    
    def apply_activation_function(self, id_layer, input_data, list_weights, list_bias):

        result = 0
        for k in self.assemble(self, id_layer, input_data, list_weights, list_bias):
            result += k
        return self.activation_function(result)

    def feedforward(self):
        pass