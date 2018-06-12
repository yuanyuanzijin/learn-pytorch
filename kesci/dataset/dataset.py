# coding:utf8
import os
from torch.utils import data
import numpy as np
import pandas as pd
import torch as t
import json

class AppData(data.Dataset):
    def __init__(self, datapath, iflabel=True):
        self.iflabel = iflabel
        self.datapath = datapath
        with open(datapath) as f:
            self.data = json.loads(f.read())

    def __getitem__(self, index):
        user_log = t.Tensor(self.data[index]['data'])
        user_log[:,2] /= 100
        user_log[:,3] /= 100
        user_log[:,4] /= 100
        user_log[:,5] /= 100
        user_log[:,6] /= 100
        user_log[:,7] /= 500
        user_log[:,8] /= 500
        user_log[:,9] /= 100
        user_log[:,10] /= 100
        user_log[:,11] /= 50
        user_log[:,12] /= 50
        user_log[:,13] /= 50
        user_source = int(self.data[index]['source'])
        one_hot = t.zeros(12)
        one_hot[user_source-1] = 1
        #user_device = self.data[index]['device']
        user_new = t.Tensor([int(self.data[index]['new'])])
        label = t.LongTensor([self.data[index]['label']])
        return user_log, t.cat([one_hot, user_new], dim=0), label

    def __len__(self):
        return len(self.data)
