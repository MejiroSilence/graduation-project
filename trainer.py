import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import softUpdate

device = torch.device("cuda")

class trainer(object):
    def __init__(self,mac,args):
        self.mse=nn.MSELoss()
        self.criticOpt=torch.optim.Adam(mac.criticParam, lr=args.criticLR)
        self.actorOpt=torch.optim.Adam(mac.actorParam, lr=args.actorLR)
        self.tau=args.tau

    def initLast(self):
        self.lastAction=None
        self.lastState=None
        self.lastObs=None

    def calculateQtot(self,n_agents,critic,mixer,obs,state,actions,lastAction,onlyMax=False):
        inputs=torch.stack([torch.cat([actions[:i].view(-1),actions[i+1:].view(-1),state,obs[i],lastAction[i],torch.tensor([i],device=device)]) for i in range(n_agents)])
        qs=critic(inputs)
        v=qs.max(1)[0]
        vTotal=torch.sum(v)
        if onlyMax:
            return vTotal
        q=qs.gather(1,actions).view(-1)
        a=q-v
        lambda_= mixer(state,actions)
        mixeda=lambda_*a
        qTotal=torch.sum(mixeda)+vTotal
        return qTotal

    def criticTrain(self,n_agents,mac,obs,state,actions,lastAction,reward,gamma):
        if self.lastState is not None:
            qTotal=self.calculateQtot(n_agents,mac.targetCritic,mac.targetMixer,obs,state,actions,lastAction,True).detach()
            qTotalLast=self.calculateQtot(n_agents,mac.evalCritic,mac.evalMixer,self.lastObs,self.lastState,lastAction,self.lastAction)
            loss=self.mse(reward+gamma*qTotal,qTotalLast)
            self.criticOpt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mac.criticParam,max_norm=10, norm_type=2)
            self.criticOpt.step()
            softUpdate(mac.targetCritic,mac.evalCritic,self.tau)
        self.lastAction=lastAction
        self.lastState=state
        self.lastObs=obs
        return
    
    def counterfactualBaseline(self,prob,q):
        return sum(prob*q)

    def actorTrain(self,mac,n_agents,actions,probs,state,obs,lastAction):
        inputs=torch.stack([torch.cat([actions[:i].view(-1),actions[i+1:].view(-1),state,obs[i],lastAction[i],torch.tensor([i],device=device)]) for i in range(n_agents)])
        qs=mac.evalCritic(inputs)
        loss=0
        for i in range(n_agents):
            a=qs[i][int(actions[i])]-self.counterfactualBaseline(probs[i],qs[i])
            loss-=a.detach()*torch.log(probs[i][int(actions[i])])
        self.actorOpt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mac.actorParam,max_norm=10, norm_type=2)
        self.actorOpt.step()
        return


            