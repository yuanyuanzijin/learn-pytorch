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


def train():
    vis = Visualizer("Kesci")
    train_data = AppData("data/train_20d_3p.json", iflabel=True, datapath2="data/test_20d_3p.json")
    val_data = AppData("data/val_20d_3p.json", iflabel=True)
    train_dataloader = DataLoader(train_data, 512, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, 256, shuffle=False, num_workers=2)

    criterion = t.nn.CrossEntropyLoss()
    learning_rate = 0.01
    weight_decay = 0.0002
    model = Sequence(14, 128, 1).cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    for epoch in range(500):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, property, label) in tqdm(enumerate(train_dataloader)):
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

        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()), lr=learning_rate))

        if loss_meter.value()[0] > previous_loss:
            learning_rate = learning_rate * 0.95
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        previous_loss = loss_meter.value()[0]

        if epoch % 20 == 19:
            model.save()


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
    train()