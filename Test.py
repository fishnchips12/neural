import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, x):
          for b,w in zip(self.biases, self.weights):
                x = sigmoid(np.dot(w,x)+b)



net = network([2,3,3])
