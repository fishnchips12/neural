import numpy as np
from PIL import Image
import os
import random

desired = []
input = []
data_base = "New_DataBase"
for i,filename in enumerate(os.listdir(data_base)):
    img = Image.open(f"{data_base}/{filename}").convert('L')  # convert image to 8-bit grayscale
    WIDTH, HEIGHT = img.size

    data = list(img.getdata())  # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    map = np.reshape(data, (100,1))
    activation = []
    for a in map:
        activation.append(a/255)
    input.append(activation)

    if len(filename) < 10:
        correct = np.zeros((10,1))
        b = filename[4]
        correct[int(b)] = 1
        desired.append(correct)

    else:
        correct = np.zeros((10,1))
        b = filename[5]
        correct[int(b)] = 1
        desired.append(correct)






def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

class network(object):
    def __init__(self):
        self.weights1 = np.random.randn(60,100)
        self.weights2 = np.random.randn(10,60)
        self.biases1 = np.random.randn(60,1)
        self.biases2 = np.random.randn(10,1)

    def feedforward(self, x):
        self.z1 = (np.dot(self.weights1, x) + self.biases1)
        self.a1 = sigmoid(self.z1)
        self.z2 = (np.dot(self.weights2, self.a1) + self.biases2)
        self.a2 = sigmoid(self.z2)
        output = self.a2
        print(x)
        return output



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
            for x,y in zip(x, y):
                a = np.reshape(x, (100, 1))
                b = np.reshape(y, (10,1))

                output = self.feedforward(a)
                self.backprop(a, b, output)


        else:
            a = np.reshape(x, (100, 1))
            b = np.reshape(y, (10, 1))
            output = self.feedforward(a)
            self.backprop(a, b, output)


net = network()


for i in range(1):
    net.train(input, desired)

net.feedforward(input[1])
