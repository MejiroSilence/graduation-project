import torch
from critic import Qnet
from mixer import qscan
from rnnAgent import gruAgent
from utils import hard_update

class pscan(object):
    def __init__(self,args):
        self.n_agents=args.agentNum
        self.agent = gruAgent(args.observeDim+2,args.agentHiddenDim,args.actionNum,args.actorLR)
        self.evalCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.targetCritic=Qnet(args.agentNum+args.stateDim+args.observeDim+1,args.criticHiddenDim,args.actionNum)
        self.evalMixer=qscan(args)
        self.targetMixer=qscan(args)
        hard_update(self.targetCritic,self.evalCritic)
        hard_update(self.targetMixer,self.evalMixer)

    def initHidden(self):
        self.hs=[self.agent.initHidden() for i in range(self.n_agents)]

        
