import torch as t
from torch import nn
from .BasicModule import BasicModule

class Sequence(BasicModule):
    def __init__(self, input_size, hidden_dim):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)


    def forward(self, input):
        out, _ = self.lstm(input)
        out = self.fc(out[:, -1, :])
        return out
