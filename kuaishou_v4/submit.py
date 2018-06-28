import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset import AppData
from torch.utils.data import DataLoader
from models.Sequence import Sequence
from torchnet import meter
import numpy as np
from tqdm import tqdm

def submit():
    model_time = "0628_10:39:34"
    load_model_path = "checkpoints/06281033/LSTM_%s.pth" % model_time
    model = Sequence(31, 128, 1).cuda()
    model.load(load_model_path)

    user_list = []
    probability = []
    submit_data = AppData("../kesci/data/data_v3_23d/submit2.json", iflabel=False)
    submit_dataloader = DataLoader(submit_data, 512, shuffle=False)
    for ii, (input, property, user_id) in tqdm(enumerate(submit_dataloader)):
        val_input = Variable(input, volatile=True).cuda()
        val_input2 = Variable(property).cuda()
        score = model(val_input, val_input2).cpu()
        probability.extend(F.sigmoid(score)[:, 0].data.tolist())
        user_list.extend(user_id.tolist())

    index = np.argmax(probability)
    print(user_list[index], probability[index])
    index2 = np.argmin(probability)
    print(user_list[index2], probability[index2])
    all = zip(probability, user_list)
    all = sorted(all, key=lambda x: x[0], reverse=True)
    getnum = 24000
    with open('../kesci/submission/submission_%s_%d.txt' % (model_time, getnum), 'w') as f:
        num = 0
        for i in all[:getnum]:
            #if i[0] > 0.5:
                f.writelines(str(i[1]) + '\n')
                num += 1
    print(num)

    with open('../kesci/submission/score_%s.txt' % model_time, 'w') as f:
        for i in all:
            f.writelines("%s\t%s\n" % (str(i[1]), str(i[0])))

if __name__ == "__main__":
    submit()