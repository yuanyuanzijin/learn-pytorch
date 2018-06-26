import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset0619 import AppData
from torch.utils.data import DataLoader
from models.BasicModule import BasicModule
from torchnet import meter
import numpy as np
from tqdm import tqdm

class Sequence(BasicModule):
    def __init__(self, input_size, hidden_dim, n_class):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 256, 1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, 128, 2, batch_first=True)
        self.lstm1_fc = nn.Linear(256, n_class)
        self.lstm2_fc = nn.Linear(128, n_class)
        self.prop_fc1 = nn.Linear(34, 24)
        self.prop_fc2 = nn.Linear(24, 1)
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(4, 6)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, input, input2):
        out1, _ = self.lstm1(input)
        out1 = self.lstm1_fc(out1[:, -1, :])
        out2, _ = self.lstm2(input)
        out2 = self.lstm2_fc(out2[:, -1, :])
        out = t.cat([out1, out2], dim=1)
        day_input = Variable(t.Tensor([[input.size(1)]] * input.size(0))).cuda()
        out = self.fc1(t.cat([out, day_input], dim=1))
        out0 = self.prop_fc2(self.prop_fc1(input2))
        out = t.cat([out, out0], dim=1)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

def submit():
    model_time = "0622_20:01:43"
    load_model_path = "checkpoints/0622_194837/LSTM_%s.pth" % model_time
    model = Sequence(15, 128, 1).cuda()
    model.load(load_model_path)

    user_list = []
    probability = []
    submit_dataloader_list = [DataLoader(AppData("data/submit_30d_change.json", iflabel=False, data_length=31-length), 32, shuffle=False, num_workers=4) for length in range(1, 31)]

    for dataloader in submit_dataloader_list:
        for ii, (input, property, user_id) in tqdm(enumerate(dataloader)):
            val_input = Variable(input, volatile=True).cuda()
            val_input2 = Variable(property).cuda()
            score = model(val_input, val_input2).cpu()
            probability.extend(t.nn.functional.softmax(score)[:, 1].data.tolist())
            user_list.extend(user_id.tolist())

    index = np.argmax(probability)
    print(user_list[index], probability[index])
    index2 = np.argmin(probability)
    print(user_list[index2], probability[index2])
    all = zip(probability, user_list)
    all = sorted(all, key=lambda x: x[0], reverse=True)
    with open('submission/submission_%s_23800.txt' % model_time, 'w') as f:
        num = 0
        for i in all[:23800]:
            #if i[0] > 0.5:
                f.writelines(str(i[1]) + '\n')
                num += 1
    print(num)

    with open('submission/score_%s_23.txt' % model_time, 'w') as f:
        for i in all:
            f.writelines("%s\t%s\n" % (str(i[1]), str(i[0])))


def test_offline():
    test_dataloader_list = [DataLoader(AppData("data/test_test.json", iflabel=True, data_length=30-length), 128, shuffle=False, num_workers=2) for length in range(1, 2)]

    model_time = "0621_22:46:40"
    load_model_path = "checkpoints/0621_224213/LSTM_%s.pth" % model_time
    model = Sequence(15, 128, 1).cuda()
    model.load(load_model_path)

    test_cm, test_f1 = val(model, test_dataloader_list)

    print(test_f1)
    print(test_cm)


def val(model, dataloader_list):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for dataloader in dataloader_list:
        for ii, data in tqdm(enumerate(dataloader)):
            input, property, label = data
            label = label.view(-1)
            val_input = Variable(input, volatile=True).cuda()
            val_input2 = Variable(property).cuda()
            score = model(val_input, val_input2)
            confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    precision = cm_value[0][0] / cm_value[0].sum()
    recall = cm_value[0][0] / cm_value[:, 0].sum()
    f1 = 2 * precision * recall / (precision + recall)
    return confusion_matrix, f1


if __name__ == "__main__":
    submit()