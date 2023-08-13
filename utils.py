import numpy as np

class activation_function:
    def __init__(self, activate, derivative):
        self.activate = activate
        self.derivative = derivative

class cost_function:
    def __init__(self, cost, derivative):
        self.cost = cost
        self.derivative = derivative

    def __call__(self, a, y):
        return self.cost(a, y)

def relu(z):
    return np.maximum(z, 0)

def relu_prime(z):
    return np.where(z > 0, 1, 0)

class ReLU(activation_function):
    def __init__(self):
        super().__init__(relu, relu_prime)

def stable_softmax(z):
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))

def stable_softmax_prime(z):
    return stable_softmax(z) * (1 - stable_softmax(z))

class Softmax(activation_function):
    def __init__(self):
        super().__init__(stable_softmax, stable_softmax_prime)

def cross_entropy(a, y):
    return -np.sum(y * np.log(a))

def cross_entropy_prime(a, y):
    return a - y

class CrossEntropy(cost_function):
    def __init__(self):
        super().__init__(cross_entropy, cross_entropy_prime)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Sigmoid(activation_function):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)