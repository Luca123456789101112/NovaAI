import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, dA):
        gradient = dA
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def update_params(self,learning_rate):
        for layer in self.layers:
            if hasattr(layer, "w"):
                layer.w -= learning_rate * layer.dw
                layer.b -= learning_rate * layer.db