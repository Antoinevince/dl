import math
import os
import sys
sys.path.append("")

from mathfunction import functions

class singleperceptron():
    def __init__(self, input, weights, bias):
        self.input = input
        self.weights = weights 
        self.bias = bias

    def train(self, loss_function, pas):
        data = self.input
        ecart = pas
        
        while ecart>=0:
            
            #initialisation des pentes des différents paramètres
            weight_slope = 0
            bias_slope = 0
            
            #calcul des dérivées partielles
            for k in range(len(data[0])):
                weight_slope += (self.weights*data[0][k] - data[1][k])*2*data[0][k]
                bias_slope += (self.weights*data[0][k] - data[1][k])*2
            
            #pondération des pentes
            weight_slope = weight_slope/len(data[0])
            bias_slope = bias_slope/len(data[0])
            
            #ajustement des coefficients
            self.weights -= weight_slope*pas
            self.bias -= bias_slope*pas
        
            c = loss_function()
            old_c = c
            
            #ajustement du pas
            pas = pas/10
            
            #calcul de la fonction de perte
            for l in range(len(data[1])):
                c+= (self.weights*data[0][l] + self.bias)**2
                
            c = c/(len(data[0]))
            
            ecart = old_c - c
            
        return (self.weights, self.bias, c)

    def neuron(self):
        return 


data = [(1, 1.01), (2, 2.004), (3, 3), (4, 5.26)]
neuron = singleperceptron(data, 1, 1)
print(neuron.train(functions.quandratic_loss, 0.1))