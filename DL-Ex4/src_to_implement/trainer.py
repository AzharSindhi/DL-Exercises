import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
# from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        out = self._model(x)
        # -calculate the loss
        loss = self._crit(out, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        pred = self._model(x)
        # return the loss and the predictions
        loss = self._crit(pred, y)
        return loss.item(), pred
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        average_loss = 0
        # iterate through the training set
        for data in self._train_dl:
            x, y = data
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss = self.train_step(x, y)
            # calculate the average loss for the epoch and return it
            average_loss += loss
        
        average_loss = average_loss / len(self._train_dl)
        print("TRAIN: average loss {:.3f}".format(
                average_loss))
        return average_loss
    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        average_test_loss = 0.0
        ypred = []
        ytrue = []
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        with t.no_grad():
            # iterate through the validation set
            for data in self._val_test_dl:
                x, y = data
                ytrue.append(y.detach().numpy().astype(float))
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                pred = pred.detach().cpu().numpy().astype(float)
                # print(y.shape, pred.shape)
                ypred.append(pred)
                average_test_loss += loss
                # save the predictions and the labels for each batch
            
            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
            
            ytrue = np.resize(ytrue,(len(ytrue),2))
            ypred = np.resize(ypred,(len(ypred),2))
            ypred = np.round(ypred, 0)
            f1score = f1_score(ytrue, ypred, average=None)
            
            average_test_loss = average_test_loss / len(self._val_test_dl)
            print("TEST: average loss and f1 score: {:.3f}, ".format(average_test_loss), f1score)
            
            # return the loss and print the calculated metrics
            return average_test_loss
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses, val_losses = [], []
        epoch = 0
        same_loss_count = 0
        prev_loss = np.inf
        temp_checkpoint=1
        
        for epoch in tqdm(range(epochs)):
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()      
            val_loss = self.val_test()      
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if (train_loss < prev_loss):
                self.save_checkpoint(temp_checkpoint)

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if (same_loss_count >= self._early_stopping_patience):
                return train_losses, val_losses
            
            if (train_loss >= prev_loss):
                same_loss_count += 1
            else:
                same_loss_count = 0
            
            prev_loss = train_loss
        
        return train_losses, val_losses
    
    # def save_onnx(self, path):
    #     pass
        
        
