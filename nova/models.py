import numpy as np

from .layers import Dropout

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

    def update_params(self,learning_rate, optimization_algorithm=None , beta=0.9, beta2=0.999, epsilon=1e-8):
        for layer in self.layers:
            if hasattr(layer, "w"):
                if optimization_algorithm == "Gradient Descent" or optimization_algorithm is None:
                    layer.w -= learning_rate * layer.dw
                    layer.b -= learning_rate * layer.db
                    
                elif optimization_algorithm == "Momentum":
                    vW = np.zeros_like(layer.w)
                    vb = np.zeros_like(layer.b)
                    vW = beta * vW + (1 - beta) * layer.dw
                    vb = beta * vb + (1 - beta) * layer.db
                    layer.w -= learning_rate * vW
                    layer.b -= learning_rate * vb

                elif optimization_algorithm == "Adam":
                    vW = np.zeros_like(layer.w)
                    vb = np.zeros_like(layer.b)
                    sW = np.zeros_like(layer.w)
                    sb = np.zeros_like(layer.b)

                    vW = beta * vW + (1 - beta) * layer.dw
                    vb = beta * vb + (1 - beta) * layer.db

                    sW = beta2 * sW + (1 - beta2) * (layer.dw ** 2) 
                    sb = beta2 * sb + (1 - beta2) * (layer.db ** 2)

                    vW_corrected = vW / (1 - beta)
                    vb_corrected = vb / (1 - beta)

                    sW_corrected = sW / (1 - beta2)
                    sb_corrected = sb / (1 - beta2)

                    layer.w -= learning_rate * vW_corrected / (np.sqrt(sW_corrected) + epsilon)
                    layer.b -= learning_rate * vb_corrected / (np.sqrt(sb_corrected) + epsilon)


    def compile(self, loss):
        self.loss = loss

    def fit(self, x, y, iterations_num, optimization_algorithm=None):
        for i in range(iterations_num):
            y_pred = self.forward(x)
            loss = self.loss.forward(y, y_pred)
            dA = self.loss.backward(y, y_pred)
            self.backward(dA)
            self.update_params(learning_rate=0.05, optimization_algorithm=optimization_algorithm)     
            if i % 1000 == 0:          
                print(i, loss)    

        return f"Final training loss: {loss}"

    def predict(self, x):
        y_pred = self.forward(x, training=False)
        return y_pred

