import torch
import torch.nn as nn
import torch.nn.functional as F

class trainer(object):
    def __init__(self,mac,args):
        self.mse=nn.MSELoss
        self.criticOpt=torch.optim.Adam(mac.evalCritic.parameters()+mac.evalMixer.parameters(), lr=args.criticLR)

    def initLast(self):
        self.lastAction=None
        self.lastState=None
        self.lastObs=None

    def calculateQtot(self,n_agents,critic,mixer,obs,state,actions,lastAction):
        inputs=torch.tensor([torch.cat([actions[:i],actions[i+1:],state,obs[i],lastAction[i]]) for i in range(n_agents)])
        qs=critic(inputs)
        q=qs.gather(1,actions)
        v=qs.max(1)[0]
        vTotal=torch.sum(v)
        a=q-v
        lambda_= mixer(state,actions)
        mixeda=lambda_*a
        qTotal=torch.sum(mixeda)+vTotal
        return qTotal

    def criticTrain(self,n_agents,mac,obs,state,actions,lastAction,reward,gamma):
        if self.lastState is not None:
            qTotal=self.calculateQtot(n_agents,mac.targetCritic,mac.targetMixer,obs,state,actions,lastAction).detach()
            qTotalLast=self.calculateQtot(n_agents,mac.evalCritic,mac.evalMixer,self.lastObs,self.lastState,lastAction,self.lastAction)
            loss=self.mse(reward+gamma*qTotal,qTotalLast)
            self.criticOpt.zero_grad()
            loss.backward()
            self.criticOpt.step()
        self.lastAction=lastAction
        self.lastState=state
        self.lastObs=obs
        return

            