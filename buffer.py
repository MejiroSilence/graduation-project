from types import SimpleNamespace as SN
import torch
import numpy as np

device = torch.device("cuda")

class buffer(object):
    def __init__(self,batchSize,args):
        self.data=SN()
        self.batchSize=batchSize
        self.index=0
        self.episodesInBuffer=0
        self.max_seq_length=0
        self.args=args
        self.data.rewards=torch.zeros((self.batchSize,args.maxSteps,1),device=device)
        self.data.states=torch.zeros((self.batchSize,args.maxSteps,args.stateDim),device=device)
        self.data.obs=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,args.observeDim),device=device)
        self.data.actions=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,1),device=device)
        self.data.availableActions=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,args.actionNum),device=device)
        self.data.actionsOnehot=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,args.actionNum),device=device)
        self.data.terminated=torch.zeros((self.batchSize,args.maxSteps),device=device)
        self.data.mask=torch.zeros((self.batchSize,args.maxSteps),device=device)

    def addEpisode(self,episode):
        self.data.rewards[self.index]=episode.data.rewards[0]
        self.data.states[self.index]=episode.data.states[0]
        self.data.obs[self.index]=episode.data.obs[0]
        self.data.actions[self.index]=episode.data.actions[0]
        self.data.availableActions[self.index]=episode.data.availableActions[0]
        self.data.actionsOnehot[self.index]=episode.data.actionsOnehot[0]
        self.data.terminated[self.index]=episode.data.terminated[0]
        self.data.mask[self.index]=episode.data.mask[0]
        self.index+=1
        if self.index == self.batchSize:
            self.index=0
        if self.episodesInBuffer<self.batchSize:
            self.episodesInBuffer+=1

    def canSample(self, batchSize):
        return self.episodesInBuffer >= batchSize

    def sample(self,batchSize):
        sampledData=buffer(batchSize,self.args)
        if self.episodesInBuffer == batchSize:
            sampledData.data.rewards=self.data.rewards[:batchSize]
            sampledData.data.states=self.data.states[:batchSize]
            sampledData.data.obs=self.data.obs[:batchSize]
            sampledData.data.actions=self.data.actions[:batchSize]
            sampledData.data.availableActions=self.data.availableActions[:batchSize]
            sampledData.data.actionsOnehot=self.data.actionsOnehot[:batchSize]
            sampledData.data.terminated=self.data.terminated[:batchSize]
            sampledData.data.mask=self.data.mask[:batchSize]
        else:
            # Uniform sampling only atm
            index = np.random.choice(self.episodesInBuffer, batchSize, replace=False)
            sampledData.data.rewards=self.data.rewards[index]
            sampledData.data.states=self.data.states[index]
            sampledData.data.obs=self.data.obs[index]
            sampledData.data.actions=self.data.actions[index]
            sampledData.data.availableActions=self.data.availableActions[index]
            sampledData.data.actionsOnehot=self.data.actionsOnehot[index]
            sampledData.data.terminated=self.data.terminated[index]
            sampledData.data.mask=self.data.mask[index]

        sampledData.max_seq_length=sampledData.max_t_filled()
        
        sampledData.data.rewards=sampledData.data.rewards[:,:sampledData.max_seq_length]
        sampledData.data.states=sampledData.data.states[:,:sampledData.max_seq_length]
        sampledData.data.obs=sampledData.data.obs[:,:sampledData.max_seq_length]
        sampledData.data.actions=sampledData.data.actions[:,:sampledData.max_seq_length]
        sampledData.data.availableActions=sampledData.data.availableActions[:,:sampledData.max_seq_length]
        sampledData.data.actionsOnehot=sampledData.data.actionsOnehot[:,:sampledData.max_seq_length]
        sampledData.data.terminated=sampledData.data.terminated[:,:sampledData.max_seq_length]
        sampledData.data.mask=sampledData.data.mask[:,:sampledData.max_seq_length]

        return sampledData

        
    def max_t_filled(self):
        return torch.sum(self.data.mask,1).max(0)[0]


