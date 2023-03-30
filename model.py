import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # (500, 200)
        sum1 = torch.sum(x, dim=0) / x.shape[0]
        x = self.fc2(x)
        x = F.relu(x)  # (500, 200)
        sum2 = torch.sum(x, dim=0) / x.shape[0]
        x = self.fc3(x)
        x = torch.sigmoid(x)  # (500, 1)
        sum3 = torch.sum(x, dim=0) / x.shape[0]
        return x, sum1, sum2, sum3

    def mask_forward(self, x, index1=None, index2=None):
        x = self.fc1(x)
        x = F.relu(x)  # (500, 200)
        if index1 is not None:
            x.index_fill_(1, index1, 0)

        x = self.fc2(x)
        x = F.relu(x)  # (500, 200)
        if index2 is not None:
            x.index_fill_(1, index2, 0)

        x = self.fc3(x)
        x = torch.sigmoid(x)  # (500, 1)
        return x

    def neuron_sum(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # (500, 200)
        sum1 = torch.sum(x, dim=0) / x.shape[0]
        x = self.fc2(x)
        x = F.relu(x)  # (500, 200)
        sum2 = torch.sum(x, dim=0) / x.shape[0]
        x = self.fc3(x)
        x = torch.sigmoid(x)
        sum3 = torch.sum(x, dim=0) / x.shape[0]
        return sum1, sum2, sum3


