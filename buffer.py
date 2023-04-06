from types import SimpleNamespace as SN
import torch

device = torch.device("cuda")

class buffer(object):
    def __init__(self,batchSize,args):
        self.data=SN()
        self.batchSize=batchSize
        self.index=0
        self.data.rewards=torch.zeros((self.batchSize,args.maxSteps,1),device=device)
        self.data.states=torch.zeros((self.batchSize,args.maxSteps,args.stateDim),device=device)
        self.data.obs=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,args.observeDim),device=device)
        self.data.actions=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,1),device=device)
        self.data.availableActions=torch.zeros((self.batchSize,args.maxSteps,args.agentNum,args.actionNum),device=device)

