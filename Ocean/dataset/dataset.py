import scipy.io as sio
import numpy as np
from torch.utils import data
import warnings
import torch as t

class Ocean(data.Dataset):
    def __init__(self, path, mode='train'):

        load_data = sio.loadmat(path)

        if mode == 'train':
            self.data = t.from_numpy(load_data['train_data']).float()
            self.label = t.from_numpy(load_data['train_label']).squeeze()
        elif mode == 'val':
            self.data = t.from_numpy(load_data['test_data']).float()[:90]
            self.label = t.from_numpy(load_data['test_label']).squeeze()[:90]
        elif mode == 'test':
            self.data = t.from_numpy(load_data['test_data']).float()[90:]
            self.label = t.from_numpy(load_data['test_label']).squeeze()[90:]
        else:
            warnings.warn("Warning: Invalid mode! Mode should be train, val or test.")

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)