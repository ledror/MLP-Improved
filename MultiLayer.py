import random
from Layer import Layer
import numpy as np
from utils import *

class MultiLayer(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layers = [Layer(sizes[i], sizes[i+1], relu, relu_prime) for i in range(len(sizes)-2)]
        self.layers.append(Layer(sizes[-2], sizes[-1], softmax, None))
        self.zs = [np.zeros(size) for size in sizes[1:]]
        self.activations = [np.zeros(size) for size in sizes]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def feedforward(self, a):
        for layer in self.layers:
            z, a = layer.forward(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        for x, y in mini_batch:
            self.backprop(x, y)
        for layer in self.layers:
            layer.update(eta/len(mini_batch))
        
        self.zero_grad()

    def backprop(self, x, y):
        self.activations[0] = x
        
        for i, layer in enumerate(self.layers):
            self.zs[i], self.activations[i+1] = layer.forward(self.activations[i])

        delta = self.layers[-1].backward_last(self.activations[-1], self.activations[-2], y)
        for l in range(2, self.num_layers):
            delta = self.layers[-l].backward(self.activations[-l-1], self.zs[-l], delta)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)