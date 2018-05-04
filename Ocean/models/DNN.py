from torch import nn
from .BasicModule import BasicModule


class DNN(BasicModule):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """

    def __init__(self):
        super(DNN, self).__init__()

        self.model_name = 'dnn'

        self.normal = nn.Sequential(
            nn.Linear(199, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        x = self.normal(x)
        return x
