import numpy as np
import torch
from torch.utils.data import Dataset

'''
Taken directly from SCARF repository
'''
class SCARFDataset(Dataset):
    def __init__(self, data, target):
        self.data = np.array(data)
        self.target = np.array(target)
        #self.columns = columns

    @property
    def features_low(self):
        return self.data.min(axis=0)

    @property
    def features_high(self):
        return self.data.max(axis=0)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index):  # Return sample and label
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.target[index], dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.data)