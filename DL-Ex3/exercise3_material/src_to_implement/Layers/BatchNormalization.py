import numpy as np

from Layers.Base import BaseLayer
from copy import deepcopy

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.mean = np.zeros(self.channels, dtype=float)
        self.var = np.zeros(self.channels, dtype=float)
        self.running_mean = np.zeros(self.channels, dtype=float)
        self.running_var = np.zeros(self.channels, dtype=float)
        self.running_avg_gamma = np.zeros(self.channels, dtype=float) + 0.8
        self.eps = 1e-10
        self.initialize(None, None)
        self._optimizer = None
    
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        opt1 = deepcopy(opt)
        opt2 = deepcopy(opt)
        self._optimizer = [opt1, opt2]

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)
    
    
    def reformat(self, tensor): 
        if len(tensor.shape) == 4:  # converting CNN to FCN by the below procedure refer slide 40
            self.Batch, self.Channel, self.M, self.N = tensor.shape
            #M and N are spatial dimensions
            tensor = tensor.reshape(self.Batch, self.Channel, self.M * self.N)
            tensor = np.transpose(tensor, (0, 2, 1))#shifting of variables
            tensor = tensor.reshape(self.Batch * self.M * self.N, self.Channel)
        else:  # converting FCN to CNN:
            tensor = tensor.reshape(self.Batch, self.M * self.N, self.Channel)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(self.Batch, self.Channel, self.M, self.N)
        return tensor
    

    def update_running_params(self):
        is_mean_empty = np.array_equal(np.zeros(self.channels), self.running_mean)
        if (is_mean_empty):
            self.running_mean = self.mean
            self.running_var = self.var
        else:
            gamma = self.running_avg_gamma
            self.running_mean = gamma * self.running_mean + (1.0 - gamma)* self.mean
            self.running_var = gamma * self.running_var + (1.0 - gamma)* self.var

    def forward_bn(self, input_tensor, axis):
        if (not self.testing_phase):
            self.mean = np.mean(input_tensor, axis=axis)
            self.var = np.var(input_tensor, axis=axis)
            self.update_running_params()
        else:
            self.mean = self.running_mean.copy()
            self.var = self.running_var.copy()
        
        return self.mean, self.var

    def forward(self, input_tensor):

        self.axis = 0
        if (len(input_tensor.shape) == 4):
            input_tensor = self.reformat(input_tensor) # 4D to 2D
            self.mode = 1
        else:
            self.mode = 0
        
        self.mean, self.var = self.forward_bn(input_tensor, self.axis)

        self.x_minus_mean = input_tensor - self.mean
        self.var_eps = self.var + self.eps
        self.stddev = np.sqrt(self.var_eps)
        self.standard = (self.x_minus_mean)/ self.stddev
        output = self.weights * self.standard + self.bias
        if (self.mode == 1):
            output = self.reformat(output)        
        return output

    def backward(self, error_tensor):
        if (self.mode==1):
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(error_tensor * self.standard, axis=self.axis)
        self.gradient_bias = np.sum(error_tensor, axis=self.axis)
        if (self._optimizer != None):
            self.weights = self._optimizer[0].calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer[1].calculate_update(self.bias, self.gradient_bias)
        
        norm_mean = self.x_minus_mean
        var_eps = self.var_eps

        gamma_err = error_tensor * self.weights
        inv_batch = 1. / error_tensor.shape[0]

        grad_var = np.sum(norm_mean * gamma_err * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=self.axis)

        sqrt_var = self.stddev #np.sqrt(var_eps)
        first = gamma_err * 1. / sqrt_var

        grad_mu_two = (grad_var * np.sum(-2. * norm_mean, keepdims=True, axis=self.axis)) * inv_batch
        grad_mu_one = np.sum(gamma_err * -1. / sqrt_var, keepdims=True, axis=self.axis)

        second = grad_var * (2. * norm_mean) * inv_batch
        grad_mu = grad_mu_two + grad_mu_one

        new_error_tensor = first + second + inv_batch * grad_mu
        if (self.mode == 1):
            new_error_tensor = self.reformat(new_error_tensor)

        return new_error_tensor