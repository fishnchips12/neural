import numpy as np

x1 = np.array(([0.5],[0.1]))
x2 = np.array(([0.3],[0.7]))
x3 = np.array(([0.2],[0.4]))
x4 = np.array(([0.7],[0.8]))
x5 = np.array(([0.2],[0.1]))
x6 = np.array(([0.8],[0.9]))
x7 = np.array(([0.5],[0.1]))
x8 = np.array(([0.5],[0.2]))
x9 = np.array(([0.9],[0.3]))

y1 = np.array(([1],[0]))
y2 = np.array(([0],[1]))
y3 = np.array(([0],[1]))
y4 = np.array(([0],[1]))
y5 = np.array(([1],[0]))
y6 = np.array(([0],[1]))
y7 = np.array(([1],[0]))
y8 = np.array(([1],[0]))
y9 = np.array(([1],[0]))

input = [x1,x2,x3,x4,x5,x6,x7,x8,x9]
desired = [y1,y2,y3,y4,y5,y6,y7,y8,y9]


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


class network(object):
    def __init__(self):
        self.weights1 = np.random.randn(20,2)
        self.weights2 = np.random.randn(2,20)
        self.biases1 = np.random.randn(20,1)
        self.biases2 = np.random.randn(2,1)

    def feedforward(self, x):
        self.z1 = np.dot(self.weights1, x) + self.biases1
        self.a1 = sigmoid(self.z1)
        self.z2 = sigmoid(np.dot(self.weights2, self.a1) + self.biases2)
        self.a2 = self.z2
        output = self.a2
        return (output)

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
        for a, b in zip(x, y):
            output = self.feedforward(a)
            self.backprop(a, b, output)

net = network()
for i in range(100):
    net.train(input, desired)


a = np.array(([0.7],[0.2]))
print(net.feedforward(a))