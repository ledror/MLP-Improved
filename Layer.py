import numpy as np

class Layer(object):
    def __init__(self, input_size, output_size, activation, activation_prime):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_prime = activation_prime
        self.weights = np.random.randn(output_size, input_size) / 10.0
        self.bias = np.random.randn(output_size, 1)
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        
    def zero_grad(self):
        self.weights_grad[:] = 0
        self.bias_grad[:] = 0

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return z, self.activation(z)

    def backward(self, a, z, delta):
        delta = delta * self.activation_prime(z)
        self.bias_grad += delta
        self.weights_grad += np.dot(delta, a.T)
        delta = np.dot(self.weights.T, delta)
        return delta

    def backward_last(self, last_activation, before_last, y):
        delta = last_activation - y
        self.bias_grad += delta
        self.weights_grad += np.dot(delta, before_last.T)
        delta = np.dot(self.weights.T, delta)
        return delta
    
    def update(self, learning_rate):
        self.weights = self.weights - learning_rate * self.weights_grad
        self.bias = self.bias - learning_rate * self.bias_grad