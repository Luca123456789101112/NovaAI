#solo aplica Z = WX + b
import numpy as np
class Dense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dw = None
        self.db = None

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w) + self.b
        return self.z
    
    def backward(self, dz):
        self.dw = (np.dot(self.x.T, dz))/self.x.shape[0]
        self.db = (np.sum(dz, axis=0, keepdims=True))/self.x.shape[0]
        dx = np.dot(dz, self.w.T) # dx = da[l-1], x = a[l-1]
        return dx

