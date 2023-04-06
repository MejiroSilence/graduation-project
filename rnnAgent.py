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
        self.softmax = nn.Softmax(dim=-1)

    def initHidden(self):
        return torch.zeros(self.hiddenSize,device=device)

    def forward(self, input, hidden):
        x = F.leaky_relu(self.fc1(input))
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
    
    def forward(self,batch,t,h):
        self.buildInput(batch,t)
    
    def chooseActions(self,batch,t,e,h):
        prob=self.forward(batch,t,h)
            
        
    def buildInput(self,batch,t):
        pass