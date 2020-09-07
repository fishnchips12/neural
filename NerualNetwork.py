import numpy as np


x = np.array(([0.2],[0.3]), dtype=float)           #a < 5 b < 5 = 0001
y = np.array(([0],[0],[0],[1]), dtype=float)
x2 = np.array(([0.7],[0.9]), dtype=float)          #a > 5 b > 5 =1000
y2 = np.array(([1],[0],[0],[0]), dtype=float)
x3 = np.array(([0.2],[0.7]), dtype=float)          #a < 5 b > 5 = 0010
y3 = np.array(([[0],[0],[1],[0]]), dtype=float)
x4 = np.array(([0.9],[0.1]), dtype=float)          #a > 5 b < 5 = 0100
y4 = np.array(([0],[1],[0],[0]), dtype=float)      #-----
x5 = np.array(([0.4],[0.7]), dtype=float)          #a < 5 b > 5 = 0010
y5 = np.array(([0],[0],[1],[0]), dtype=float)
x6 = np.array(([0.8],[0.9]), dtype=float)          #a > 5 b > 5 = 1000
y6 = np.array(([1],[0],[0],[0]), dtype=float)
x7 = np.array(([0.2],[0.1]), dtype=float)          #a < 5 b < 5 = 0001
y7 = np.array(([0],[0],[0],[1]), dtype=float)
x8 = np.array(([0.4],[0.8]), dtype=float)          #a < 5 b > 5 = 0010
y8 = np.array(([0],[0],[1],[0]), dtype=float)
x9 = np.array(([0.3],[0.3]), dtype=float)          #a < 5 b < 5 = 0001
y9 = np.array(([0],[0],[0],[1]), dtype=float)
x10 = np.array(([0.6],[0.7]), dtype=float)         #a > 5 b > 5 = 1000
y10 = np.array(([1],[0],[0],[0]), dtype=float)



data = [x,x2,x3,x4,x5,x6,x7,x8,x9,x10]
desired = [y,y2,y3,y4,y5,y6,y7,y8,y9,y10]



def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


class network(object):
    def __init__(self):
        self.weights1 = np.random.randn(200,2)
        self.weights2 = np.random.randn(4,200)
        self.biases1 = np.random.randn(200,1)
        self.biases2 = np.random.randn(4,1)




    def feedforward(self, x):
        self.z1 = np.dot(self.weights1, x) + self.biases1 #3,1
        self.a1 = sigmoid(self.z1) #3,1
        self.z2 = np.dot(self.weights2, self.a1) + self.biases2 #4,1
        self.a2 = sigmoid(self.z2) #4,1
        return self.a2


    def backprop(self, x, y, output):
        self.output_error = (output - y)**2
        self.output_error_delta = 2 * (output - y) * sigmoid_prime(self.z2)
        self.layer_2_delta = np.multiply(np.dot(np.transpose(self.weights2), self.output_error_delta), sigmoid_prime(self.z1))
        self.delta_weights2 = np.dot(self.output_error_delta, np.transpose(self.a1))
        self.delta_weights1 = np.dot(self.layer_2_delta, np.transpose(x))
        self.weights2 = self.weights2 - self.delta_weights2
        self.weights1 = self.weights1 - self.delta_weights1
        self.biases2 = self.biases2 - self.output_error_delta
        self.biases1 = self.biases1 - self.layer_2_delta




    def train(self, x, y):
        if len(x) > 2:
            for x,y in zip(data, desired):
                output = self.feedforward(x)
                self.backprop(x, y, output)
        else:
            output = self.feedforward(x)
            self.backprop(x, y, output)

net = network()

for i in range(50):
    net.train(data,desired)

a = np.array(([0.2],[0.1]),dtype=float)

print(net.feedforward(a))

