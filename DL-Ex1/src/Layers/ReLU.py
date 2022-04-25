import numpy as np

from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.tensor_value = None

    def forward(self, input_tensor):
        updated_input = np.maximum(0, input_tensor) 
        self.tensor_value = updated_input
        return updated_input

    def backward(self, error_tensor):
        updated_input = self.tensor_value
        #where the input is less or equal to zero replace with zero, else replace with tensor value
        output = np.where(updated_input<=0, 0, error_tensor)
        return output




