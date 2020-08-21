from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        #self.dropout1 = nn.Dropout2d(0.25)
        #self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128 * 1 * 10, 128) # 20 kmers/2
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        #print(x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #print(x.size())
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        #print(x.size())
        x = F.max_pool2d(x, kernel_size=(1,2))
        #print(x.size())
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        #print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        #print(x.size())
        #x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x.size())
        #print(x)
        output = F.softmax(x, dim=1)
        #print(output.size())
        #print(output)
        return output