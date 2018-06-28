import torch as t
from torch import nn
import torch.nn.functional as F
from models.BasicModule import BasicModule

class Sequence(BasicModule):
    def __init__(self, input_size, hidden_dim, n_class):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, 3, batch_first=True)
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