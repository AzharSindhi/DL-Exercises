import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weights):
        norm_output = self.alpha * np.linalg.norm(weights.flatten(), ord=2)**2
        #print("norm: ",norm_output)
        return norm_output

    def calculate_gradient(self, weights):
        output = self.alpha*weights
        return output


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def norm(self, weights):
        norm_output = self.alpha*np.linalg.norm(weights.flatten(), ord=1)
        return norm_output
        
    def calculate_gradient(self, weights):
        output = self.alpha * np.sign(weights)
        return output

