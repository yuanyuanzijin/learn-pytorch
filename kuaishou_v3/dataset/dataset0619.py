# coding:utf8
import os
from torch.utils import data
import numpy as np
import pandas as pd
import torch as t
import json

class AppData(data.Dataset):
    def __init__(self, datapath, iflabel=True, data_length=1):
        self.iflabel = iflabel
        self.datapath = datapath
        with open(datapath) as f:
            self.data = json.loads(f.read())[str(data_length)]


    def __getitem__(self, index):
        data = self.data[index]['data']

        user_log = t.FloatTensor(data)
        user_log[:,2] = (user_log[:,2] - 5.2) / 733
        user_log[:,3] = (user_log[:,3] - 0.9) / 26
        user_log[:,4] = (user_log[:,4] - 0.7) / 28
        user_log[:,5] = (user_log[:,5] - 4.3) / 726
        user_log[:,6] = (user_log[:,6] - 0.05) / 0.7
        user_log[:,7] = (user_log[:,7] - 10.5) / 2270
        user_log[:,8] = (user_log[:,8] - 6.0) / 786
        user_log[:,9] = (user_log[:,9] - 0.3) / 9.7
        user_log[:,10] = (user_log[:,10] - 0.1) / 1.0
        user_log[:,11] = (user_log[:,11] - 0.03) / 0.2
        user_log[:,12] = (user_log[:,12] - 0.0) / 0.0003
        user_log[:,13] = (user_log[:,13] - 0.0) / 0.01
        user_log[:,14] = (user_log[:,14] - 0.07) / 0.9
        user_source = int(self.data[index]['source'])
        one_hot_source = t.zeros(12)
        one_hot_source[user_source-1] = 1
        device_type = int(self.data[index]['device'])
        one_hot_device = t.zeros(22)
        if device_type <= 20:
            one_hot_device[device_type] = 1
        else:
            one_hot_device[21] = 1
        one_hot = t.cat([one_hot_source, one_hot_device], dim=0)
        #days = t.FloatTensor([len(self.data[index]['data'])])
        #user_new = t.Tensor([int(self.data[index]['new'])])
        #properties = t.cat([one_hot, active_percent], dim=0)

        if self.iflabel:
            label = t.LongTensor([self.data[index]['label']])
            return user_log, one_hot, label
        else:
            return user_log, one_hot, self.data[index]['user_id']

    def __len__(self):
        return len(self.data)
