import numpy as np
import random
from Layer import Layer
from utils import cost_function
from utils import ReLU, Softmax, CrossEntropy, Sigmoid

class MultiLayer(object):
    def __init__(self, layers, cost: cost_function):
        self.num_layers = len(layers)
        self.layers = layers
        self.cost = cost

    # default constructor:
    #   ReLU for hidden layers
    #   Softmax for output layers
    #   CrossEntropy for cost function
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.layers = [Layer(sizes[i], sizes[i + 1], ReLU()) for i in range(self.num_layers - 1)]
        self.layers[-1].activation = Softmax()
        self.cost = CrossEntropy()

    def feedforward(self, input):
        a = input
        for layer in self.layers:
            _, a = layer.forward(a)
        return a

    def backpropagation(self, x, y):
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers]
        a = x
        zs = []
        activations = [a]
        for layer in self.layers:
            z, a = layer.forward(a)
            zs.append(z)
            activations.append(a)
        # delta_L
        # delta = self.cost.derivative(activations[-1], y) * self.layers[-1].activation.derivative(zs[-1])
        delta = activations[-1] - y
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            delta = self.layers[-l+1].backward(zs[-l], delta)
            nabla_w[-l] = delta @ activations[-l - 1].T
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def SGD(self, data, epochs, learning_rate, batch_size=16, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(data)
        for epoch in range(epochs):
            random.shuffle(data)
            batches = [data[k:k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

    def update_batch(self, batch, learning_rate):
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers]
        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        for layer, nw, nb in zip(self.layers, nabla_w, nabla_b):
            layer.update(nw, nb, learning_rate / len(batch))

    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in results)