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
        data = np.array(self.data[index]['data'], dtype=np.float32)
        data = np.delete(data, [22,23,24,25,26], axis=1)
        user_log = t.from_numpy(data)
        user_log[:,2] = (user_log[:,2] - 0.162) / 0.136
        user_log[:,4] = (user_log[:,4] - 0.023) / 0.111
        user_log[:,6] = (user_log[:,6] - 13.3) / 3356
        user_log[:,7] = (user_log[:,7] - 6.04) / 786
        user_log[:,8] = (user_log[:,8] - 12.6) / 3262
        user_log[:,9] = (user_log[:,9]) / 14.0
        user_log[:,10] = (user_log[:,10]) / 1.91
        user_log[:,11] = (user_log[:,11] - 0.03) / 0.293
        user_log[:,12] = (user_log[:,12] - 0.0001) / 0.0003
        user_log[:,13] = (user_log[:,13] - 0.0006) / 0.012
        user_log[:,14] = (user_log[:,14]) / 9.64
        user_log[:,15] = (user_log[:,15] - 6.17) / 961
        user_log[:,16] = (user_log[:,16] - 1.29) / 64.0
        user_log[:,17] = (user_log[:,17]) / 43.1
        user_log[:,18] = (user_log[:,18] - 4.90) / 941
        user_log[:,19] = (user_log[:,19]) / 2.55

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
        #active_percent = t.FloatTensor([self.data[index]['active_percent']])
        #user_new = t.Tensor([int(self.data[index]['new'])])
        #properties = t.cat([one_hot, active_percent], dim=0)
        #times_to_label = [0, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 1.0]
        if self.iflabel:
            times = self.data[index]['target']
            # if times > 0.5:
            #     target = 1
            # else:
            #     target = 0
            #target = times_to_label[times]
            target = t.FloatTensor([times])
            return user_log, one_hot, target
        else:
            return user_log, one_hot, self.data[index]['user_id']

    def __len__(self):
        return len(self.data)

