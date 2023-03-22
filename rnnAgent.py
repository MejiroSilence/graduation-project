import torch
import torch.nn as nn
import torch.nn.functional as F

class gruAgent(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize,lr):
        super(gruAgent, self).__init__()
        self.hiddenSize = hiddenSize
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.gru = nn.GRUCell(hiddenSize, hiddenSize)
        self.fc2=nn.Linear(hiddenSize, outputSize)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)  

    def initHidden(self):
        return torch.zeros(1, 1, self.hiddenSize)

    def forward(self, input, hidden):
        x = F.relu(self.fc1(input))
        h = self.gru(x, hidden)
        q = self.fc2(h)
        return q, h