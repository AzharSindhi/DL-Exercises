import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None

    def forward(self,prediction_tensor, label_tensor):
        self.prediction = prediction_tensor
        prediction_tensor = np.where(label_tensor==1, prediction_tensor + np.finfo(float).eps, label_tensor)
        new_tensor = []
        new_tensor.append(prediction_tensor[prediction_tensor>0])
        loss = np.sum(-np.log(new_tensor))
        
        return loss

    def backward(self, Label_tensor):
        output = np.divide(-Label_tensor, self.prediction)
        return output

         
