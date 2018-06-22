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
            print(len(self.data))
        elif mode == 'val':
            self.data = t.from_numpy(load_data['test_data']).float()[45:135]
            self.label = t.from_numpy(load_data['test_label']).squeeze()[45:135]
        elif mode == 'test':
            test_data = t.from_numpy(load_data['test_data']).float()
            test_label = t.from_numpy(load_data['test_label']).squeeze()
            self.data = test_data[0:45].append(test_data[135:])
            self.label = test_label[0:45].append(test_label[135:])
        else:
            warnings.warn("Warning: Invalid mode! Mode should be train, val or test.")

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)