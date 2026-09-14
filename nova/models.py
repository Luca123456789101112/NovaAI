import numpy as np

from layers import Dropout

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x, training=True):
        output = x
        for layer in self.layers:
            if isinstance(layer, Dropout):
                output = layer.forward(output, training)
            else:
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

    def compile(self, loss):
        self.loss = loss

    def fit(self, x, y, iterations_num):
        for i in range(iterations_num):
            y_pred = self.forward(x)
            loss = self.loss.forward(y, y_pred)
            dA = self.loss.backward(y, y_pred)
            self.backward(dA)
            self.update_params(learning_rate=0.05)        

        return f"Final training loss: {loss}"

    def predict(self, x):
        y_pred = self.forward(x, training=False)
        return y_pred

