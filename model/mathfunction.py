import math

def Relu(x):
    return 0.5*(x+abs(x))

def logistic(x):
    denom = (1+ math.exp(x))**(-1)
    return (math.exp(x))*denom


