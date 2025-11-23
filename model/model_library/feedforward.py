import model.mathfunction as mathfunction
import math


class MultiLayerPerceptron():

    def __init__(self, input_weights, input_biases, hidden_weights, hidden_biases, output_weights, output_biases):

        self.input_weights = input_weights
        self.input_biases = input_biases

        self.hidden_wheights = hidden_weights
        self.hidden_biases = hidden_biases

        self.output_weights = output_weights
        self.output_biases = output_biases



        
    def neuron(self, input_values, id):
        """single neuron implemented"""

        sum = input_values[id]*self.input_weights[id] + self.input_biases[id]
        new_value = mathfunction.Relu(sum)

        return new_value
    
    def train(self):

        """
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
        """

        pass

      


        



