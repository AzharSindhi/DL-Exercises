import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv("data.csv", sep=';')
# print(data.head())
train_data, val_data = train_test_split(data, test_size=0.02, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

# hyperparameters
batch_size = 8
epochs = 30
early_stopping_patience = 3
num_workers = 4
shuffle = True

lr = 0.01
mom = 0.9
# optimizer = 

train_dataset = ChallengeDataset(train_data, "train")
train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle = shuffle, num_workers=num_workers)

val_dataset = ChallengeDataset(val_data, "val")
val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=1,
                                            shuffle = False, num_workers=num_workers)

# create an instance of our ResNet model
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=mom)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, criterion, optimizer, train_dataloader, val_dataloader, 
                    cuda=True, early_stopping_patience=early_stopping_patience)

# go, go, go... call fit on trainer
res = trainer.fit(epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')