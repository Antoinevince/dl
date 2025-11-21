import math

def Relu(x):
    """
    x: a random real number
    """
    return 0.5*(x+abs(x))

def logistic(x):
    """
    x: a random real number
    """
    denom = (1+ math.exp(x))**(-1)
    return (math.exp(x))*denom

def gudermannian(x):
    """
    x: a random real number
    """
    return 2*(math.atan(math.tanh(x/2)))

def algebraic(x, k):
    """
    x: a strictly positive real number
    k: a random real number which is the parameter of the algebraic function taking x as an argument
    """
    denom = (1+math.abs(x)**k)**(-(1/k))
    return x*denom


