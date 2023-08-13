import numpy as np
from Layer import Layer
from utils import cost_function
from utils import ReLU, Softmax, CrossEntropy

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
        self.layers = [Layer(sizes[i], sizes[i + 1], ReLU()) for i in range(self.num_layers - 2)]
        self.layers.append(Layer(sizes[-2], sizes[-1], Softmax()))
        self.cost = CrossEntropy()

    def feedforward(self, input):
        a = input
        for layer in self.layers:
            _, a = layer.forward(a)
        return a

    def backpropagation(self, input, target):
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers]
        a = input
        zs = []
        activations = [a]
        for layer in self.layers:
            z, a = layer.forward(a)
            zs.append(z)
            activations.append(a)
        # delta_L
        delta = self.cost.derivative(activations[-1], target) * self.layers[-1].activation.derivative(zs[-1])
        nabla_w[-1] = delta @ activations[-2].T
        nabla_b[-1] = delta
        for l in range(2, self.num_layers):
            delta = self.layers[-l + 1].backward(zs[-l], delta)
            nabla_w[-l] = delta @ activations[-l - 1].T
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def SGD(self, data, epochs, learning_rate, batch_size=16):
        n = len(data)
        for epoch in range(epochs):
            random.shuffle(data)
            batches = [data[k:k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            print("Epoch {0}: {1}%".format(epoch+1, self.evaluate(data) / n * 100))

    def update_batch(self, batch, learning_rate):
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.layers]
        for input, target in batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(input, target)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        for layer, nw, nb in zip(self.layers, nabla_w, nabla_b):
            layer.update(nw, nb, learning_rate)

    def evaluate(self, data):
        results = [(np.argmax(self.feedforward(input)), target) for input, target in data]
        return sum(int(x == y) for x, y in results)



