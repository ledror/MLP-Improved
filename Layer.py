import random
import numpy as np
from utils import activation_function

class Layer(object):
    def __init__(self, input_size, output_size, activation: activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.bias = np.random.randn(output_size, 1)
        
    def forward(self, x):
        z = self.weights @ x + self.bias
        return z, self.activation.activate(z)

    def backward(self, z, delta):
        return (self.weights.T @ delta) * self.activation.derivative(z)
    
    def update(self, nabla_w, nabla_b, learning_rate):
        self.weights -= learning_rate * nabla_w
        self.bias -= learning_rate * nabla_b
