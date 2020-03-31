import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import deque

Actions = 4  # 游戏的四种行为/八种行为


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.replay_memory = deque()
        self.actions = Actions

        self.conv1 = nn.Conv2d(4, 32, kernel_size=12, stride=2, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=12, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=12, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3)

        self.fc1 = nn.Linear(278400, 256)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(256, self.actions)

    def forward(self, input):
        out = self.net(input)
        out = out.view(out.size()[0], -1)
        # print("-->{}".format(out.size()))
        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        return out


