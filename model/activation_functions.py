import math

    
#from here: activation functions
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


def identity(x):
        return x