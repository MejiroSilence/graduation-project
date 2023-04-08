from utils import onehot
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

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
            nn.Softmax(dim=-1)
        )
        self.comb=torch.tensor(list(combinations([i for i in range(agentNum)],2)),device=device)
        self.comb0=self.comb[:,0].unsqueeze(-1)
        self.comb1=self.comb[:,1].unsqueeze(-1)
        self.combSize=int(agentNum*(agentNum-1)/2)
    
    def forward(self,batch,t=None):
        inputs=self.buildInput(batch,t)
        outs=self.fc(inputs)#b*t*comb*2
        lambda_=torch.zeros((batch.batchSize,batch.max_seq_length if t is None else 1,self.agentNum),device=device)
        for i in range(self.agentNum):
            for j in range(self.agentNum):
                if i==j:
                    continue
                if i<j:
                    lambda_[:,:,i]+=outs[:,:,int((2*self.agentNum-1-i)*i/2+j-i-1),0]
                else:
                    lambda_[:,:,i]+=outs[:,:,int((2*self.agentNum-1-j)*j/2+i-j-1),1]
        lambda_=lambda_*2/(self.agentNum-1)
        return lambda_

    
    def buildInput(self,batch,t=None):
        bs = batch.batchSize
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        #state
        inputs.append(batch.data.states[:, ts].unsqueeze(2).repeat(1, 1, self.combSize, 1))
        #agent
        agent1 = self.comb0.new(*self.comb0.shape[:-1], self.agentNum).zero_()
        agent1.scatter_(-1, self.comb0.long(), 1)
        inputs.append(agent1.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        agent2 = self.comb1.new(*self.comb1.shape[:-1], self.agentNum).zero_()
        agent2.scatter_(-1, self.comb1.long(), 1)
        inputs.append(agent2.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        #action 
        actions=batch.data.actionsOnehot[:,ts]
        actions1=actions.index_select(dim=2,index=self.comb0.squeeze())
        actions2=actions.index_select(dim=2,index=self.comb1.squeeze())
        inputs.append(actions1)
        inputs.append(actions2)

        inputs = torch.cat([x.reshape(bs, max_t, self.combSize, -1) for x in inputs], dim=-1)
        return inputs

class qscan(nn.Module):
    def __init__(self, args):
        super(qscan, self).__init__()
        self.args=args
        self.embed=nn.Sequential(
            nn.Linear(args.stateDim+args.agentNum+args.actionNum,args.mixerHiddenDim),
            nn.LeakyReLU(),
            nn.Linear(args.mixerHiddenDim, args.mixerHiddenDim),
            nn.LeakyReLU(),
            nn.Linear(args.mixerHiddenDim, args.embedSize),
        )
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args.embedSize, nhead=args.nhead)
        self.encoder_norm = torch.nn.LayerNorm(args.embedSize)
        self.encoder=torch.nn.TransformerEncoder(self.encoder_layer, args.attentionLayer, self.encoder_norm)
        self.calLambda_=nn.Sequential(
            nn.Linear(args.embedSize,args.mixerHiddenDim),
            nn.LeakyReLU(),
            nn.Linear(args.mixerHiddenDim, args.mixerHiddenDim),
            nn.LeakyReLU(),
            nn.Linear(args.mixerHiddenDim, 1),
        )

    def forward(self,batch,t=None):
        inputs=self.buildInput(batch,t)# batch*t*agent*(state+agent+action)
        inputs=self.embed(inputs)
        inputs=self.encoder(inputs)
        lambda_=self.calLambda_(inputs)
        lambda_=lambda_.reshape(batch.batchSize,-1,self.args.agentNum)
        lambda_=lambda_/lambda_.mean(dim=-1,keepdim=True)
        return lambda_
    
    def buildInput(self,batch,t=None):
        bs = batch.batchSize
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch.data.states[:, ts].unsqueeze(2).repeat(1, 1, self.args.agentNum, 1))
        # action
        actions=batch.data.actionsOnehot[:,ts]#b t a action
        inputs.append(actions)

        inputs.append(torch.eye(self.args.agentNum, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = torch.cat([x.reshape(bs, max_t, self.args.agentNum, -1) for x in inputs], dim=-1)
        return inputs.reshape(bs*max_t,self.args.agentNum,-1)
        