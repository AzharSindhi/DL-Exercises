from cProfile import label
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode

        self._transform = {'train': tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                           tv.transforms.Normalize(mean=train_mean, std=train_std)]),
                           'val': tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(), 
                                                         tv.transforms.Normalize(mean=train_mean, std=train_std)])
                           }

    def __len__(self):
        #print("Data length: ", len(self.data))
        return len(self.data)

    def __getitem__(self, index):
        path_img = "./"
        img_path = Path(path_img, self.data.iloc[index, 0])
        # print("image_path:", img_path)

        img = imread(img_path)
        label = self.data.iloc[index, [1,2]].to_numpy(dtype=int)
        #print(type(img))
        #print(type(label))

        img = gray2rgb(img)

        if self._transform:
            img = self._transform[self.mode](img)

        #print(type(img))

        label = torch.tensor(label, dtype=torch.float)

        return img, label


