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
        self.lstm = nn.LSTM(input_size, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_class)
        self.fc2 = nn.Linear(12, 32)
        self.fc3 = nn.Linear(32, 1)
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
    model_time = "0619_19:06:20"
    load_model_path = "checkpoints/0619_184154/LSTM_%s.pth" % model_time
    model = Sequence(15, 128, 1).cuda()
    model.load(load_model_path)

    user_list = []
    probability = []
    submit_dataloader_list = [DataLoader(AppData("data/submit_23d_change.json", iflabel=False, data_length=24-length), 256, shuffle=False, num_workers=4) for length in range(1, 24)]

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
    with open('submission/submission_%s_23800_23.txt' % model_time, 'w') as f:
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
    test_offline()