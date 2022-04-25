from argparse import OPTIONAL
from Optimization.Constraints import L1_Regularizer, L2_Regularizer
import numpy as np

class Optimizer:
    def __init__(self, regularizer = None):
        #super().__init__()
        self.regularizer = regularizer

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        reg_term = 0
        if (self.regularizer != None):
            reg_term = self.learning_rate* self.regularizer.calculate_gradient(weight_tensor)

        updated_weight = weight_tensor - self.learning_rate*(gradient_tensor) - reg_term
        return updated_weight


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.new_gradient = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        reg_term = 0
        if (self.regularizer != None):
            reg_term = self.learning_rate* self.regularizer.calculate_gradient(weight_tensor)

        self.new_gradient = (self.momentum_rate*(self.new_gradient)) - self.learning_rate*(gradient_tensor)
        updated_weight = weight_tensor + self.new_gradient - reg_term
        return updated_weight


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.iter_k = 1
        self.v_k = 0
        self.r_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        
        self.v_k = (self.mu*(self.v_k)) + (1-self.mu)*gradient_tensor
        self.r_k = (self.rho*(self.r_k)) + (1-self.rho)*(gradient_tensor**2)

        #bias correction
        v = self.v_k/(1-(np.power(self.mu,self.iter_k))) 
        r = self.r_k/(1-(np.power(self.rho,self.iter_k))) 
        self.iter_k = self.iter_k + 1

        #weight update
        reg_term = 0
        if (self.regularizer != None):
            reg_term = self.learning_rate* self.regularizer.calculate_gradient(weight_tensor)

        updated_weight = weight_tensor - self.learning_rate*v/(np.sqrt(r) + np.finfo(float).eps) - reg_term

        return updated_weight
