import numpy as np

class Sigmoid:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.act_output = 1/(1 + np.exp(-input_tensor))
        return self.act_output

    def backward(self, error_tensor):
        #error_tensor = self.forward(error_tensor)

        new_output = self.act_output*(1-self.act_output)
        return new_output* error_tensor
