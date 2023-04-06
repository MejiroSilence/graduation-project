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

    def initHidden(self,batchSize):
        self.hs=self.agent.initHidden().unsqueeze(0).expand(batchSize, self.n_agents, -1)

    def forward(self,batch,t):
        inputs=self.buildInput(batch,t)
        q, self.hs = self.agent(inputs, self.hs)

        q = torch.nn.functional.softmax(q, dim=-1)
            
        return q.view(batch.batchSize, self.n_agents, -1)

    def chooseActions(self,batch,t,e):
        probs=self.forward(batch,t)
        availableActions=batch.data.availableActions[:,t]
        #debug
        print(availableActions.size())
        print(probs.size())
        input()
        probs=probs*availableActions
        probs=probs/(torch.sum(probs,dim=-1,keepdim=True)+ 1e-8)
        actionsNum = (availableActions.sum(-1, keepdim=True) + 1e-8)
        probs=(1-e)*probs+e/actionsNum*availableActions

        pickedActions = torch.distributions.Categorical(probs).sample().long()

        print("actions size")
        print(pickedActions.size())
        input()

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

        
