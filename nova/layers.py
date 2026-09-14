#solo aplica Z = WX + b
import numpy as np
class Dense:
    def __init__(self, input_dim, output_dim, initializer, l2_lambda=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_lambda = l2_lambda
        if initializer == "he":
            self.w = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
            self.b = np.zeros((1, output_dim))
        elif initializer == "zeros":
            self.w = np.zeros((input_dim, output_dim))
            self.b = np.zeros((1, output_dim))
        elif initializer == "random":
            self.w = np.random.randn(input_dim, output_dim)
            self.b = np.zeros((1, output_dim))
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w) + self.b
        return self.z
    
    def backward(self, dz):
        self.dw = (np.dot(self.x.T, dz))/self.x.shape[0] + (self.l2_lambda/self.x.shape[0] * self.w)
        self.db = (np.sum(dz, axis=0, keepdims=True))/self.x.shape[0]
        dx = np.dot(dz, self.w.T) # dx = da[l-1], x = a[l-1]
        return dx

class Dropout:
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) < self.keep_prob) / self.keep_prob
            return x * self.mask
        else:
            return x

    def backward(self, dA):
        return dA * self.mask