import torch as t
from torch import nn
import torch.nn.functional as F
from models.BasicModule import BasicModule

class DoubleSequence(BasicModule):
    def __init__(self, input_size, hidden_dim, n_class):
        super(DoubleSequence, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_dim, 2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size, hidden_dim, 2, batch_first=True)
        self.lstm1_fc = nn.Linear(hidden_dim, n_class)
        self.lstm2_fc = nn.Linear(hidden_dim, n_class)
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc_all = nn.Linear(3, 6)
        self.fc_all2 = nn.Linear(6, 1)


    def forward(self, input, input2):
        out1, _ = self.lstm1(input)
        out1 = self.lstm1_fc(out1[:, -1, :])
        out2, _ = self.lstm2(input[:, -7:, :])
        out2 = self.lstm2_fc(out2[:, -1, :])
        out3 = F.relu(self.fc1(input2))
        out3 = F.relu(self.fc2(out3))
        out = t.cat([out1, out2], dim=1)
        out = t.cat([out, out3], dim=1)
        out = F.relu(self.fc_all(out))
        out = self.fc_all2(out)
        return out