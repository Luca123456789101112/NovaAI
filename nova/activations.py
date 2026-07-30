import numpy as np

class ReLu:
    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        return np.maximum(0, self.z)

    def backward(self, dA):
        dz = np.multiply(dA, np.where(self.z > 0, 1, 0))
        return dz

class Softmax:
    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0, keepdims=True)
    def backward(self, dA):
        s = self.forward(self.z)
        dz = np.multiply(dA, s * (1-s))
        return dz

class Tanh: 
    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        return np.tanh(z)

    def backward(self, dA):
        dz = np.multiply(dA, 1 - np.tanh(self.z)**2)
        return dz

class Sigmoid: 
    def __init__(self):
        self.z = None

    def forward(self, z):
        self.z = z
        return 1 / (1 + np.exp(-z))

    def backward(self, dA):
        s = self.forward(self.z)
        dz = np.multiply(dA, s * (1 - s))
        return dz