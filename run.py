from pscan import pscan
from smac.env import StarCraft2Env
import torch
import numpy as np

def train(args,epoch,epoch_len,env_args):
    sc_env = StarCraft2Env(**env_args)
    env_info = sc_env.get_env_info()
    mac = pscan(args)
    for epoch_i in range(epoch):
        t=0
        sc_env.reset()
        terminated = False
        mac.initHidden()
        while not terminated: #TODO: add max step
            obs = sc_env.get_obs()
            state=sc_env.get_state()
            lastAction=torch.zeros(env_info.n_agents)
            actions=torch.zeros(env_info.n_agents)
            for i in range(env_info.n_agents):
                q,h=mac.agents[i](torch.cat([obs[i],lastAction]),mac.hs[i])
                actionMask=sc_env.get_avail_agent_actions(i)
                action,prob=mac.agents[i].chooseAction(q,actionMask,args.epsilon)
                mac.hs[i]=h
                actions[i]=action
            reward, terminated, _ = sc_env.step(actions)
            t+=1
            criticTrain(env_info,mac,obs,state,actions,lastAction,reward,terminated)
            lastAction=actions
        sc_env.close()

def criticTrain(env_info,mac,obs,state,actions,lastAction,reward,terminated):
    inputs=torch.tensor([torch.cat([actions[:i],actions[i+1:],state,obs[i],lastAction[i]]) for i in range(env_info.n_agents)])
    qs=mac.evalCritic(inputs)
    q=qs.gather(1,actions)
    v=qs.max(1)[0]
    vTotal=torch.sum(v)
    a=q-v
    lambda_= mac.mixer(state,actions)
    mixeda=lambda_*a
    qTotal=torch.sum(mixeda)+vTotal
    
