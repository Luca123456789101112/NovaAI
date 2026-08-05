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
        self.a = None

    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.a

    def backward(self, dA):
        s = self.a
        dot = np.sum(s * dA, axis=1, keepdims=True)
        dz = s * (dA - dot)
        return dz

class Tanh:

    def __init__(self):
        self.a = None

    def forward(self,z):
        self.a = np.tanh(z)
        return self.a

    def backward(self,dA):
        return dA * (1 - self.a**2)

class Sigmoid:

    def __init__(self):
        self.a = None


    def forward(self,z):
        self.a = 1/(1+np.exp(-z))
        return self.a


    def backward(self,dA):
        return dA * self.a * (1-self.a)