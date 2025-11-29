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
        pass

      


        



