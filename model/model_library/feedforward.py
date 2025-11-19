import torch
import torch.nn


class FeedForwardNeuralNetwork(torch.nn.Module):

    def __init__(self):
        super(FeedForwardNeuralNetwork,self).__init__()
        self.l1 = torch.nn.Linear()
        self.l2 = torch.nn.Linear()
        self.relu = torch.nn.ReLU()
    
    def linear(self):
        pass 