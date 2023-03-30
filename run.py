from pscan import pscan
from smac.env import StarCraft2Env
import torch
import numpy as np
from trainer import trainer

def train(args,epoch,epoch_len,env_args):
    sc_env = StarCraft2Env(**env_args)
    env_info = sc_env.get_env_info()
    mac = pscan(args)
    epochTrainer=trainer(mac,args)
    for epoch_i in range(epoch):
        t=0
        sc_env.reset()
        epochTrainer.initLast()
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
            epochTrainer.criticTrain(env_info.n_agents,mac,obs,state,actions,lastAction,reward,args.gamma)
            lastAction=actions
        sc_env.close()




    
