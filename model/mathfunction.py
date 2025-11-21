import torch
import torch.nn


def Relu():
    pass

def sigmoid():
    pass







def normalizationlayer(vecteur):

    mean = torch.mean(vecteur)

    var = torch.var(vecteur, unbiased= True)

    sigma = torch.std(vecteur, unbiased = True)

    pass


