import numpy as np

class BinaryCrossEntropy:
    def __init__(self,y_true,y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def forward(self):
        m = self.y_true.shape[0]

        loss = -(np.sum(np.multiply(self.y_true, np.log(self.y_pred) + np.multiply((1-self.y_true), np.log(1-self.y_pred)))))/m

        return loss

    def backward(self):
        m = self.y_true.shape[0]
        da = (-(self.y_true/self.y_pred) + (1-self.y_true)/(1-self.y_pred))/m
        return da
