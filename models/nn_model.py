import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 1024)
        self.fc8 = nn.Linear(1024, 1024)
        self.fc9 = nn.Linear(1024, 1  )

        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x1 = self.dp1(F.relu(self.fc1(x)))
        x2 = self.dp2(F.relu(self.fc2(x1)))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
        x5 = F.relu(self.fc5(x4)) + x3
        x6 = F.relu(self.fc6(x5)) + x2
        x7 = F.relu(self.fc7(x6)) + x1
        x8 = F.relu(self.fc8(x7))
        x9 = self.fc9(x8)

        return x9
