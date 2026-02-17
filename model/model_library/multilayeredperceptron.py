import math
import random
import numpy
import sys
import regex
import os
import scipy

#sys.path.append("/Users/dossierantoine/Documents/GitHub/dl/model/mathfunction.py")
sys.path.insert(0, os.path.abspath('model'))


import activation_functions
import cost_functions
import derivative





class FeedForwardNeuralNetwork():

    def __init__(self, list_depth_layer, activation_function):
        """
        list_depth_layer : int list -> list containing the depth of each layer

        inner_layers = int 

        number_hidden_layers = int

        hidden_layers =int list 2-uplet list list -> it is a list containing, for each layer, a list containg a 2-uplet of int lists respectively for weights and biases of each neuron

        outter_layers: int
        """
        self.list_depth_layer = list_depth_layer

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
        for k in self.assemble(id_layer, input_data, list_weights, list_bias):
            result += k
        return self.activation_function(result)


   #""" def distribute(self, input, id_layer):
       # for k in range(len(self.list_depth_layer[id_layer])):
            #pass"""

    def feedforward(self, initial_data, total_list_weights, total_list_bias):

        data = initial_data
        
        for k in range(self.total_neuron_number()-1):
            data = self.apply_activation_function(1, initial_data, total_list_weights[k], total_list_bias[k])

        return data
    

    def total_neuron_number(self):
        
        total_neuron_number = 0
        for k in self.list_depth_layer:
            total_neuron_number += k
        return total_neuron_number


    def sgd(self, training_data, epoch, learning_rate, cost_function):
        
        """
        training_data : int list list
        """

        #temporarily set to 0
        total_neuron_number = self.total_neuron_number()

        random_weights = [random.random() for k in range(total_neuron_number)]
        random_biases = [random.random() for k in range(total_neuron_number)]

        

        for k in range(epoch):
            random.shuffle(training_data[0]), random.shuffle(training_data[1])
            
            error_vect = [cost_function(training_data[0][k], self.feedforward(training_data[1], random_weights, random_biases)[k], len(training_data)) for k in range(len(training_data))]
            error_value = 0

            
            error_gradient = numpy.gradient(error_vect)
            print(error_gradient)

            for k in range(len(random_weights)):
                random_weights[k] -= learning_rate*error_gradient
                random_biases[k] -= learning_rate*error_gradient

        return (random_weights, random_biases)
    
    


ffn_example = FeedForwardNeuralNetwork([1], activation_functions.identity)
print(ffn_example.sgd([[1, 2, 3, 5, 6], [3, 6.05, 9.83, 16, 19.47]], 50000, 0.001, cost_functions.quadratic_loss))