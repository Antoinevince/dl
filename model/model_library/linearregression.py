import math
import mathfunction


class singleperceptron():
    def __init__(self, input):
        self.input = input

    def train(self, data, loss_function, pas):
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
        pass