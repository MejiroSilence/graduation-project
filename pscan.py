import torch
from critic import Qnet
from mixer import qpair
from rnnAgent import gruAgent
from utils import hardUpdate

class pscan(object):
    def __init__(self,args):
        self.n_agents=args.agentNum
        self.agent = gruAgent(args.observeDim+args.agentNum+args.actionNum,args.agentHiddenDim,args.actionNum,args.actorLR).cuda()
        self.evalCritic=Qnet(args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.targetCritic=Qnet(args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.evalMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.targetMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.criticParam=list(self.evalCritic.parameters())+list(self.evalMixer.parameters())
        self.actorParam=self.agent.parameters()
        hardUpdate(self.targetCritic,self.evalCritic)
        hardUpdate(self.targetMixer,self.evalMixer)

    def initHidden(self):
        self.hs=[self.agent.initHidden() for i in range(self.n_agents)]

        
