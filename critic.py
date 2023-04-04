import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self,intputSize,hiddenSize,outputSize):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(intputSize, hiddenSize)
        self.fc2=nn.Linear(hiddenSize, hiddenSize)
        self.fc3=nn.Linear(hiddenSize, outputSize)
    
    def forward(self, input):
        x=F.leaky_relu(self.fc1(input))
        x=F.leaky_relu(self.fc2(x))
        x=self.fc3(x)
        return x