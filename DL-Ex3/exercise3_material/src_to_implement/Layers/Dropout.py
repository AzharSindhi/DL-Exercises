import numpy as np
from Layers.Base import BaseLayer

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):
        if self.testing_phase == False:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability
            return input_tensor * self.mask
        else:
            return input_tensor

    def backward(self, error_tensor):
        if self.testing_phase == False:
            return error_tensor*self.mask

        else:
            return error_tensor

