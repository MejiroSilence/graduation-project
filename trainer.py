from mixer import qpair
from critic import Qnet
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hardUpdate, softUpdate,onehot

device = torch.device("cuda")

class trainer(object):
    def __init__(self,mac,args):
        self.mse=nn.MSELoss()
        self.tau=args.tau
        self.actionNum=args.actionNum
        self.n_agents=args.n_agents
        self.evalCritic=Qnet(args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.targetCritic=Qnet(args.agentNum*args.actionNum+args.stateDim+args.observeDim+args.agentNum,args.criticHiddenDim,args.actionNum).cuda()
        self.evalMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.targetMixer=qpair(args.stateDim,args.mixerHiddenDim,args.agentNum,args.actionNum).cuda()
        self.criticParam=list(self.evalCritic.parameters())+list(self.evalMixer.parameters())
        hardUpdate(self.targetCritic,self.evalCritic)
        hardUpdate(self.targetMixer,self.evalMixer)
        self.criticOpt=torch.optim.Adam(mac.criticParam, lr=args.criticLR)
        self.actorOpt=torch.optim.Adam(mac.actorParam, lr=args.actorLR)

    def initLast(self):
        self.lastAction=None
        self.lastState=None
        self.lastObs=None

    def calculateQtot(self,n_agents,critic,mixer,obs,state,actions,lastAction,onlyMax=False):
        inputs=self.buildCriticInput(actions,state,obs,lastAction,n_agents)
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
            loss=self.mse(10*reward+gamma*qTotal,qTotalLast)
            self.criticOpt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mac.criticParam,max_norm=10, norm_type=2)
            self.criticOpt.step()
            softUpdate(mac.targetCritic,mac.evalCritic,self.tau)
            softUpdate(mac.targetMixer,mac.evalMixer,self.tau)
        self.lastAction=lastAction
        self.lastState=state
        self.lastObs=obs
        return
    
    def counterfactualBaseline(self,prob,q):
        return sum(prob*q)
    
    def buildCriticInput(self,actions,state,obs,lastAction,n_agents):
        return torch.stack([torch.cat([torch.cat([onehot(lastAction[j],self.actionNum) if i==j else onehot(actions[j],self.actionNum)  for j in range(n_agents)]),state,obs[i],onehot(i,n_agents)]) for i in range(n_agents)])

    def actorTrain(self,mac,n_agents,actions,probs,state,obs,lastAction):
        inputs=self.buildCriticInput(actions,state,obs,lastAction,n_agents)
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
    
    def getQtotal(self,qs,actions):
        v=qs.max(3)[0]
        vTotal=torch.sum(v,dim=2)
        taken = torch.gather(qs, dim=3, index=actions).squeeze(3)
        print("mix size check")
        print(taken.size())
        print(v.size())
        input()
        a=taken-v
        lambda_=mixer()


    
    def _train_critic(self, batch, rewards, terminated, actions, mask, bs):
        target_q_vals = self.targetCritic(batch,self.n_agents)[:, :]
        targets_taken = torch.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        targrtixedQ

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_q_vals)[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        for t in reversed(range(rewards.size(1))):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue

            q_t = self.critic(batch, t)
            q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
            targets_t = targets[:, t]

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log
    
    def train(self,batch):
        bs = batch.batcdSize
        rewards=batch.data.rewards
        actions=batch.data.actions
        terminated=batch.data.terminated.float()
        mask=batch.data.mask.float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        availableActions=batch.data.availableActions
        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions,
                                                        critic_mask, bs)


            