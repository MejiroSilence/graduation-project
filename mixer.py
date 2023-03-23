import torch
import torch.nn as nn
import torch.nn.functional as F

class qpair(nn.Module):
    def __init__(self, stateDim,hiddenSize,agentNum):
        super(qpair, self).__init__()
        self.agentNum=agentNum
        self.fcs=[[nn.Sequential(
            nn.Linear(4+stateDim, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize, 2),
            nn.Softmax()
        ) for j in range(i+1,agentNum)] for i in range(agentNum)]#for agent i, j,fcs[i][j-i-1] is the linear layer for them


    def forward(self,state,actions):
        g=[[self.fcs[i][j-i-1](torch.cat((state,torch.tensor(i),torch.tensor(j),actions[i],actions[j]),0)) for j in range(i+1,self.agentNum)]for i in range(self.agentNum)]
        lambda_=[]
        for i in range(self.agentNum):
            sum=0
            for j in range(self.agentNum):
                if i==j:
                    continue
                if i>j:
                    sum+=g[j][i-j-1]
                else:
                    sum+=g[i][j-i-1]
            lambda_.append(sum)
        return lambda_

class qscan(nn.Module):
    def __init__(self, args):
        super(qscan, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, q1, q2):
        q1 = self.dropout(q1)
        q2 = self.dropout(q2)
        q1 = self.fc1(q1)
        q2 = self.fc1(q2)
        q1 = self.fc2(q1)
        q2 = self.fc2(q2)
        q1 = self.fc3(q1)
        q2 = self.fc3(q2)
        q1 = q1.squeeze()
        q2 = q2.squeeze()
        return q1, q2