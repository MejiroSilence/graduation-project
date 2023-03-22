import torch
from critic import Qnet
from rnnAgent import gruAgent
from utils import hard_update

"""
args:
    agentNum
    observeDim
    actionNum
    agentHiddenDim
    actorLR
    stateDim
    criticHiddenDim
    criticLR
"""
class pscan(object):
    def __init__(self,args):
        self.agents = [gruAgent(args.observeDim+2,args.agentHiddenDim,args.actionNum,args.actorLR) for i in range(args.agentNum)]
        self.evalCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.targetCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.criticOptimizer = torch.optim.Adam(self.evalCritic.parameters(), lr=args.criticLR)
        hard_update(self.targetCritic,self.evalCritic)
