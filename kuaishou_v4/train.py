import torch as t
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from dataset.dataset import AppData
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from models.BasicModule import BasicModule
from tqdm import tqdm
import time

class Sequence(BasicModule):
    def __init__(self, input_size, hidden_dim, n_class):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, 2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_class)
        self.fc2 = nn.Linear(34, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(2, 6)
        self.fc5 = nn.Linear(6, 1)


    def forward(self, input, input2):
        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])
        out2 = F.relu(self.fc2(input2))
        out2 = F.relu(self.fc3(out2))
        out = t.cat([out, out2], dim=1)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

def train():
    vis = Visualizer("Kesci" + time.strftime('%m%d%H%M'))
    train_data = AppData("../kesci/data/data_v3_23d/train.json", iflabel=True)
    val_data = AppData("../kesci/data/data_v3_23d/val.json", iflabel=True)
    train_dataloader = DataLoader(train_data, 128, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, 512, shuffle=False, num_workers=2)
    test_data = AppData("../kesci/data/data_v3_23d/test.json", iflabel=True)
    test_dataloader = DataLoader(test_data, 512, shuffle=False, num_workers=2)

    criterion = t.nn.BCEWithLogitsLoss().cuda()
    learning_rate = 0.002
    weight_decay = 0.0003
    model = Sequence(28, 128, 1).cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    for epoch in range(500):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, property, target) in tqdm(enumerate(train_dataloader)):
            input = Variable(data).cuda()
            input2 = Variable(property).cuda()
            target = Variable(target).cuda()
            output = model(input, input2)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data[0])

            if ii % 100 == 99:
                vis.plot('loss', loss_meter.value()[0])

        if epoch % 3 == 2:
            train_cm, train_f1 = val(model, train_dataloader)
            vis.plot('train_f1', train_f1)
        val_cm, val_f1 = val(model, val_dataloader)

        vis.plot_many({
            'val_f1': val_f1,
            'learning_rate': learning_rate
        })
        if loss_meter.value()[0] > previous_loss:
            learning_rate = learning_rate * 0.95
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        previous_loss = loss_meter.value()[0]

        if epoch % 3 == 2:
            model.save()
            test_cm, test_f1 = val(model, test_dataloader)
            vis.plot('test_f1', test_f1)
            vis.log("训练集：{train_f1:%}, {train_pre:%}, {train_rec:%} | 验证集：{val_f1:%}, {val_pre:%}, {val_rec:%} | \
            测试集：{test_f1:%}, {test_pre:%}, {test_rec:%} | {train_true_num:%}, {val_true_num:%}, {test_true_num:%}".format(
                train_f1=train_f1, val_f1=val_f1, test_f1=test_f1,
                train_true_num=train_cm.value()[:, 0].sum()/len(train_data),val_true_num=val_cm.value()[:, 0].sum()/len(val_data),test_true_num=test_cm.value()[:, 0].sum()/len(test_data),
                train_pre=train_cm.value()[0][0] / train_cm.value()[0].sum(), train_rec=train_cm.value()[0][0] / train_cm.value()[:, 0].sum(),
                val_pre=val_cm.value()[0][0] / val_cm.value()[0].sum(), val_rec=val_cm.value()[0][0] / val_cm.value()[:, 0].sum(),
                test_pre=test_cm.value()[0][0] / test_cm.value()[0].sum(), test_rec=test_cm.value()[0][0] / test_cm.value()[:, 0].sum()
            ))


def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, property, target = data
        target = target.view(-1)
        val_input = Variable(input, volatile=True).cuda()
        val_input2 = Variable(property).cuda()
        score = F.sigmoid(model(val_input, val_input2).data)
        score[score >= 0.45] = 1
        score[score < 0.45] = 0
        score_op = 1 - score
        target[target >= 0.5] = 1
        target[target < 0.5] = 0
        confusion_matrix.add(t.cat([score_op, score], dim=1), target)
        #confusion_matrix.add(score.data.squeeze(), target.type(t.FloatTensor))

    model.train()
    cm_value = confusion_matrix.value()
    precision = cm_value[0][0] / cm_value[0].sum()
    recall = cm_value[0][0] / cm_value[:, 0].sum()
    f1 = 2 * precision * recall / (precision + recall)
    return confusion_matrix, f1


if __name__ == "__main__":
    train()