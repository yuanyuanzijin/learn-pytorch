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
        self.fc2 = nn.Linear(13, 32)
        self.fc3 = nn.Linear(32, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, input, input2):
        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])
        out = t.cat([out, input2], dim=1)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

def submit():
    load_model_path = "checkpoints/LSTM_0615_22:42:44.pth"
    model = Sequence(14, 128, 1).cuda()
    model.load(load_model_path)

    user_list = []
    probability = []
    submit_data = AppData("data/submit_data_20d.json", iflabel=False)
    submit_dataloader = DataLoader(submit_data, 512, shuffle=False)
    for ii, (input, property, user_id) in tqdm(enumerate(submit_dataloader)):
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
    with open('submission/submission_06_15_22_45_21785.txt', 'w') as f:
        for i in all[:21785]:
            f.writelines(str(i[1]) + '\n')
    print(all[21785], all[23800])
    print(len(all))


def test_offline():
    test_data = AppData("data/train_20d_3p.json", iflabel=True)
    test_dataloader = DataLoader(test_data, 128, shuffle=False, num_workers=2)

    load_model_path = "checkpoints/LSTM_0616_08:30:43.pth"
    model = Sequence(14, 128, 1).cuda()
    model.load(load_model_path)

    test_cm, test_f1 = val(model, test_dataloader)

    print(test_f1)
    print(test_cm)


def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
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