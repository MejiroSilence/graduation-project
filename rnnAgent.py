import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class gruAgent(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize, lr):
        super(gruAgent, self).__init__()
        self.hiddenSize = hiddenSize
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.gru = nn.GRUCell(hiddenSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, outputSize)

    def initHidden(self):
    # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hiddenSize).zero_()

    def forward(self, inputs, hidden_state=None):
        b, a, e = inputs.size()
        
        x = F.leaky_relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.hiddenSize)
        h = self.gru(x, hidden_state)
        q = self.fc2(h)
        return q.view(b, a, -1), h.view(b, a, -1)
