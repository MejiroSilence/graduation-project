import torch
from critic import Qnet
from mixer import qpair
from rnnAgent import gruAgent
import numpy as np
from epsilon_schedules import DecayThenFlatSchedule

class pscan(object):
    def __init__(self,args):
        self.n_agents=args.agentNum
        self.agent = gruAgent(args.observeDim+args.agentNum+args.actionNum,args.agentHiddenDim,args.actionNum,args.actorLR).cuda()
        self.actorParam=self.agent.parameters()
        self.epi_scheduler= DecayThenFlatSchedule(args.startE,args.finishE,args.epsilonAnnealTime,decay="linear")

    def initHidden(self,batchSize):
        self.hs=self.agent.initHidden().unsqueeze(0).expand(batchSize, self.n_agents, -1)

    def forward(self,batch,t):
        inputs=self.buildInput(batch,t)
        q, self.hs = self.agent(inputs, self.hs)

        #q = q/max(q.max(),abs(q.min()))

        q = torch.nn.functional.softmax(q, dim=-1)
            
        return q.view(batch.batchSize, self.n_agents, -1)

    def chooseActions(self,batch,t,t_env):
        probs=self.forward(batch,t)
        availableActions=batch.data.availableActions[:,t]
        e=self.epi_scheduler.eval(t_env)
        if np.random.uniform() > e:
            probs=probs*availableActions
            pickedActions = torch.distributions.Categorical(probs).sample().long()
        else:
            pickedActions=torch.distributions.Categorical(availableActions).sample().long()

        return pickedActions

        
    def buildInput(self,batch,t):
        bs = batch.batchSize
        inputs = []
        inputs.append(batch.data.obs[:, t])  # b1av
        if t == 0:
            inputs.append(torch.zeros_like(batch.data.actionsOnehot[:, t]))
        else:
            inputs.append(batch.data.actionsOnehot[:, t-1])
        inputs.append(torch.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

        
