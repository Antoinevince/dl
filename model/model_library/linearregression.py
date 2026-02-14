import math
import random
import numpy
import sys
import regex
import os

import scipy




#sys.path.append("/Users/dossierantoine/Documents/GitHub/dl/model/mathfunction.py")
sys.path.insert(0, os.path.abspath('model'))


from model.activation_functions import two_variables_function
from derivative import derivative_from_list_of_values



class SinglePerceptron():
    def __init__(self, file_directory):
        """
        
        """

        #self.weight = weight 
        #self.bias = bias

        self.file_dir = file_directory

    #function that aims to get the bias and weights
    def get_coefs(self):
        with open(f"{self.file_dir}", "r") as file:
            content = file.read()

            if content == "":
                raise ValueError
            else:
                #récupérer les poids cractères par caractères en faisant attention au /
                coefs = regex.split("/", content)
                float_coefs = []
                for k in coefs:
                    float_coefs.append(float(k))
                
                return float_coefs
                

    def output(self, input):
        weight = self.get_coefs()[0]
        bias = self.get_coefs()[1]
        return (weight*input +bias)
    
    def training(self, learning_rate, number_epoch, cost_function, training_data, batch_size):
       #not working
        """
        learning_rate: int
        number_epoch: int
        cost_function = function from mathfunctions
        training_data: int list list
        batch_size = int
        """

        self.weight = random.random()
        self.bias = random.random()

        
        with open(f"{self.file_dir}", "w") as creating_file:
            creating_file.write(f"{self.weight}/{self.bias}")
        

        for k in range(number_epoch):

           

            grad = derivative_from_list_of_values(training_data)

            new_weights = self.weight
            new_bias = self.bias

            new_weights -= grad
            new_bias -= grad

        

        print(self.weight, self.bias)

#writing the weights and bias in a file
        with open(f"{self.file_dir}", "w") as file:
            char = f"{str(self.weight)}/{str(self.bias)}"
            file.write(char)


        return self.weight, self.bias
