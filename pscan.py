import torch
from critic import Qnet
from mixer import qscan
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
    epsilon
"""
class pscan(object):
    def __init__(self,args):
        self.agents = [gruAgent(args.observeDim+1,args.agentHiddenDim,args.actionNum,args.actorLR) for i in range(args.agentNum)]
        self.evalCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.targetCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.mixer=qscan(args)
        hard_update(self.targetCritic,self.evalCritic)

    def initHidden(self):
        self.hs=[agent.initHidden() for agent in self.agents]

        
