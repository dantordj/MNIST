from numpy import *
from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z


    g = zeros(z.shape)
    # =========================== TODO ==================================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z.
    sig = sigmoid(z)
    g = array([y * (1 - y) for y in sig])

    return g




