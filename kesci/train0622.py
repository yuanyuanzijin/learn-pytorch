import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset0619 import AppData
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from models.BasicModule import BasicModule
from tqdm import tqdm
import time

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


def train():
    vis = Visualizer("Kesci")
    train_dataloader_list = [DataLoader(AppData("data/train_23d_change.json", iflabel=True, data_length=24-length), 64, shuffle=True, num_workers=4) for length in range(1, 24)]
    val_dataloader_list = [DataLoader(AppData("data/val_23d_change.json", iflabel=True, data_length=24-length), 128, shuffle=False, num_workers=2) for length in range(1, 24)]
    test_dataloader_list = [DataLoader(AppData("data/test_23d_change.json", iflabel=True, data_length=24-length), 128, shuffle=False, num_workers=2) for length in range(1, 24)]

    criterion = t.nn.CrossEntropyLoss(weight=t.Tensor([1, 1.15])).cuda()
    learning_rate = 0.01
    weight_decay = 0.00012
    model = Sequence(15, 128, 1).cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    for epoch in range(500):
        loss_meter.reset()
        confusion_matrix.reset()

        for dataloader in train_dataloader_list:
            for ii, (data, property, label) in tqdm(enumerate(dataloader)):
                input = Variable(data).cuda()
                input2 = Variable(property).cuda()
                target = Variable(label).cuda().view(-1)
                output = model(input, input2)

                optimizer.zero_grad()
                loss = criterion(output, target) 
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.data[0])
                confusion_matrix.add(output.data, target.data)

            #if ii % 100 == 99:
            vis.plot('loss', loss_meter.value()[0])

        if epoch % 3 == 2:
            train_cm, train_f1 = val(model, train_dataloader_list)
            vis.plot('train_f1', train_f1)
        val_cm, val_f1 = val(model, val_dataloader_list)

        vis.plot_many({
            'val_f1': val_f1,
            'learning_rate': learning_rate
        })

        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
        #     train_cm=str(confusion_matrix.value()), lr=learning_rate))

        if loss_meter.value()[0] > previous_loss:
            learning_rate = learning_rate * 0.95
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        previous_loss = loss_meter.value()[0]

        if epoch % 3 == 2:
            model.save()
            test_cm, test_f1 = val(model, test_dataloader_list)
            vis.plot('test_f1', test_f1)

            vis.log("{train_f1}, {val_f1}, {test_f1}, model:{model}, {train_cm}, {val_cm}, {test_cm}".format(
                train_f1=train_f1, val_f1=val_f1, test_f1=test_f1, model=time.strftime('%m%d %H:%M:%S'),
                train_cm=str(train_cm.value()[0][1]/train_cm.value()[1][0]), val_cm=str(val_cm.value()[0][1]/val_cm.value()[1][0]), test_cm=str(test_cm.value()[0][1]/test_cm.value()[1][0])))


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
    train()