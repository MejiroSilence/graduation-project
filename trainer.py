from mixer import qpair
from critic import Qnet
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_td_lambda_targets, hardUpdate, softUpdate,onehot

device = torch.device("cuda")

class trainer(object):
    def __init__(self,mac,args):
        self.mse=nn.MSELoss()
        self.tau=args.tau
        self.actionNum=args.actionNum
        self.n_agents=args.agentNum
        self.entropy=args.entropy
        self.mac=mac
        self.args=args
        self.evalCritic=Qnet(2*args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.targetCritic=Qnet(2*args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.evalMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.targetMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.criticParam=list(self.evalCritic.parameters())+list(self.evalMixer.parameters())
        hardUpdate(self.targetCritic,self.evalCritic)
        hardUpdate(self.targetMixer,self.evalMixer)
        self.criticOpt=torch.optim.Adam(self.criticParam, lr=args.criticLR)
        self.actorOpt=torch.optim.Adam(mac.actorParam, lr=args.actorLR)
    
    def getQtotal(self,qs,actions,lambda_):
        v=qs.max(3)[0]
        vTotal=torch.sum(v,dim=2)
        taken = torch.gather(qs, dim=3, index=actions.unsqueeze(-1)).squeeze(3)
        a=taken-v
        mixedA=lambda_*a
        aTotal=torch.sum(mixedA,dim=2)
        qTotal=aTotal+vTotal
        return qTotal
    
    def _train_critic(self, batch, rewards, terminated, actions, mask, bs):
        target_q_vals = self.targetCritic(batch,self.n_agents,self.actionNum)
        targrtLabmda=self.targetMixer(batch)
        targrtQTot=self.getQtotal(target_q_vals,actions,targrtLabmda)

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targrtQTot, self.args.gamma, self.args.td_lambda)

        q_vals = torch.zeros_like(target_q_vals)[:,:-1]

        for t in reversed(range(rewards.size(1))):
            #mask_t = mask[:, t].expand(-1, self.n_agents)
            mask_t = mask[:, t]
            if mask_t.sum() == 0:
                continue

            q_t = self.evalCritic(batch,self.n_agents,self.actionNum, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.actionNum)
            lambda_t=self.evalMixer(batch,t)
            evalQTot=self.getQtotal(q_t,actions[:,t].unsqueeze(1),lambda_t)

            targets_t = targets[:, t]

            td_error = (evalQTot - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.criticOpt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.criticParam,max_norm=10, norm_type=2)
            self.criticOpt.step()

        return q_vals
    
    def train(self,batch):
        bs = batch.batchSize
        rewards=batch.data.rewards[:,:-1]
        actions=batch.data.actions
        terminated=batch.data.terminated[:,:-1].float()
        mask=batch.data.mask[:,:-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        availableActions=batch.data.availableActions[:,:-1]
        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals = self._train_critic(batch, rewards, terminated, actions,critic_mask, bs)

        actions=actions[:,:-1]
        mac_out = []
        self.mac.initHidden(batch.batchSize)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[availableActions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)

        # Calculated baseline
        q_vals = q_vals.reshape(-1, self.actionNum)
        pi = mac_out.view(-1, self.actionNum)
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = torch.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken = torch.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.actorOpt.zero_grad()
        loss = coma_loss 
        loss.backward()
        nn.utils.clip_grad_norm_(self.mac.actorParam,max_norm=10, norm_type=2)
        self.actorOpt.step()

        hardUpdate(self.targetCritic,self.evalCritic)
        hardUpdate(self.targetMixer,self.evalMixer)


            