import math
import os


from mathfunction import functions

class singleperceptron():
    def __init__(self, input, weights):
        self.input = input
        self.weights = weights 

    def train(self, loss_function, pas):
        data = self.input
        while ecart>=0:
            
            #initialisation des pentes des différents paramètres
            weight_slope = 0
            bias_slope = 0
            
            #calcul des dérivées partielles
            for k in range(len(data[0])):
                weight_slope += (a*data[0][k] - data[1][k])*2*data[0][k]
                bias_slope += (a*data[0][k] - data[1][k])*2
            
            #pondération des pentes
            weight_slope = weight_slope/len(data[0])
            bias_slope = bias_slope/len(data[0])
            
            #ajustement des coefficients
            a -= weight_slope*pas
            b -= bias_slope*pas
        
            old_c = c
            
            #ajustement du pas
            #pas = pas/10
            
            #calcul de la fonction de perte
            for l in range(len(data[1])):
                c+= (a*data[0][l] + b)**2
                
            c = c/(len(data[0]))
            
            ecart = old_c - c
            
        return (a, b, c)

    def neuron(self):
        return 


data = [(1, 1.01), (2, 2.004), (3, 3), (4, 5.26)]
neuron = singleperceptron(data, 0)
print(neuron.train(functions.quandratic_loss, 0.1))