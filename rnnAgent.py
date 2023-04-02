import torch
import torch.nn as nn
import torch.nn.functional as F


class gruAgent(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, lr):
        super(gruAgent, self).__init__()
        self.hiddenSize = hiddenSize
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.gru = nn.GRUCell(hiddenSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)
        self.softmax = nn.Softmax(dim=-1)

    def initHidden(self):
        return torch.zeros(self.hiddenSize)

    def forward(self, input, hidden):
        x = F.relu(self.fc1(input))
        h = self.gru(x, hidden)
        q = self.fc2(h)
        return q, h

    def chooseAction(self, q, actionMask,e):
        prob = self.softmax(q)
        prob = prob * actionMask
        prob = prob / torch.sum(prob)
        availableNum = sum(actionMask)
        for i in range(len(actionMask)):
            if actionMask[i]:
                prob[i] =(1-e) * prob[i] + e / availableNum
        action =  torch.distributions.Categorical(prob).sample().detach()
        return action,prob
        