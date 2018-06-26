import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset import AppData
from torch.utils.data import DataLoader
from models.BasicModule import BasicModule
from torchnet import meter
import numpy as np
from tqdm import tqdm

class Sequence(BasicModule):
    def __init__(self, input_size, hidden_dim, n_class):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_class)
        self.fc2 = nn.Linear(34, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(2, 6)
        self.fc5 = nn.Linear(6, 2)

    def forward(self, input, input2):
        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])
        out2 = F.relu(self.fc2(input2))
        out2 = F.relu(self.fc3(out2))
        out = t.cat([out, out2], dim=1)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

def submit():
    model_time = "0626_18:18:16"
    load_model_path = "checkpoints/06261810/LSTM_%s.pth" % model_time
    model = Sequence(28, 128, 1).cuda()
    model.load(load_model_path)

    user_list = []
    probability = []
    submit_data = AppData("../kesci/data/data_v3_23d/submit.json", iflabel=False)
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
    with open('../kesci/submission/submission_%s_23800.txt' % model_time, 'w') as f:
        num = 0
        for i in all[:23800]:
            #if i[0] > 0.5:
                f.writelines(str(i[1]) + '\n')
                num += 1
    print(num)

    with open('../kesci/submission/score_%s.txt' % model_time, 'w') as f:
        for i in all:
            f.writelines("%s\t%s\n" % (str(i[1]), str(i[0])))

if __name__ == "__main__":
    submit()