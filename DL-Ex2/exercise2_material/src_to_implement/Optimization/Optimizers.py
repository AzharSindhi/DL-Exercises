import numpy as np

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - self.learning_rate*(gradient_tensor)
        return updated_weight


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.new_gradient = 0
        #self.new_gradient = np.zeros((1,1))

    def calculate_update(self, weight_tensor, gradient_tensor):

        self.new_gradient = (self.momentum_rate*(self.new_gradient)) - self.learning_rate*(gradient_tensor)
        #self.new_gradient = (self.momentum_rate*(self.new_gradient))[:,None] - self.learning_rate*(gradient_tensor)
        updated_weight = weight_tensor + self.new_gradient
        return updated_weight


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.iter_k = 1
        self.v_k = 0
        self.r_k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        #first order and second order gradient descent

        self.v_k = (self.mu*(self.v_k)) + (1-self.mu)*gradient_tensor
        self.r_k = (self.rho*(self.r_k)) + (1-self.rho)*(gradient_tensor**2)

        #bias correction
        v = self.v_k/(1-(np.power(self.mu,self.iter_k))) 
        r = self.r_k/(1-(np.power(self.rho,self.iter_k))) 
        self.iter_k = self.iter_k + 1

        #weight update
        updated_weight = weight_tensor - self.learning_rate*v/(np.sqrt(r) + np.finfo(float).eps)

        return updated_weight
