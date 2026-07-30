import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        m = self.y_true.shape[0]
        loss = -(np.sum(np.multiply(self.y_true, np.log(self.y_pred) + np.multiply((1-self.y_true), np.log(1-self.y_pred)))))/m
        return loss

    def backward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        m = self.y_true.shape[0]
        da = (-(self.y_true/self.y_pred) + (1-self.y_true)/(1-self.y_pred))/m
        return da
