from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = []
    for r in range(m):
        y_layer = append(array([1]), X[r,:])
        for i in range(num_layers-1):
            y_layer = dot(Theta[i], y_layer)
            y_layer = sigmoid(y_layer)
            y_layer = append(array([1]), y_layer)
        y_layer = y_layer[1:]
        cand = 0
        max_cand = 0
        for i in range(len(y_layer)):
            if y_layer[i] > max_cand:
                cand = i
                max_cand = y_layer[i]
        p += [cand]
 
    return p

