from utils import onehot
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class qpair(nn.Module):
    def __init__(self, stateDim,hiddenSize,agentNum,actionNum):
        super(qpair, self).__init__()
        self.agentNum=agentNum
        self.actionNum=actionNum
        self.fc=nn.Sequential(
            nn.Linear(stateDim+2*agentNum+2*actionNum, hiddenSize),
            nn.LeakyReLU(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.LeakyReLU(),
            nn.Linear(hiddenSize, 2),
            nn.Softmax()
        )

    def forward(self,state,actions):
        g=[[self.fc(torch.cat([state,onehot(i,self.agentNum),onehot(j,self.agentNum),onehot(actions[i],self.actionNum),onehot(actions[j],self.actionNum)],0)) for j in range(i+1,self.agentNum)]for i in range(self.agentNum)]
        lambda_=[]
        for i in range(self.agentNum):
            sum=0
            for j in range(self.agentNum):
                if i==j:
                    continue
                if i>j:
                    sum+=g[j][i-j-1][1]
                else:
                    sum+=g[i][j-i-1][0]
            lambda_.append(sum)
        lambda_=torch.tensor(lambda_,device=device)
        lambda_=lambda_/torch.mean(lambda_)
        return lambda_

#TODO: these code below are generated by copilot, and need to be modified
class qscan(nn.Module):
    def __init__(self, args):
        pass