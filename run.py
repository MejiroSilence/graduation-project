from pscan import pscan
from smac.env import StarCraft2Env
import torch
import numpy as np
from trainer import trainer


print("cuda status: ",torch.cuda.is_available())
device = torch.device("cuda")

def train(args):
    sc_env = StarCraft2Env()
    env_info = sc_env.get_env_info()
    args.agentNum=env_info["n_agents"]
    args.observeDim=env_info["obs_shape"]
    args.actionNum=env_info["n_actions"]
    args.stateDim=env_info["state_shape"]
    mac = pscan(args)
    epochTrainer=trainer(mac,args)
    for epoch_i in range(args.epoch):
        t=0
        sc_env.reset()
        epochTrainer.initLast()
        terminated = False
        mac.initHidden()
        while not terminated: #TODO: add max step
            obs = torch.tensor(np.array(sc_env.get_obs()))
            state=torch.tensor(sc_env.get_state())
            lastAction=torch.zeros(env_info["n_agents"],dtype=torch.int64).unsqueeze(-1)
            actions=torch.zeros(env_info["n_agents"],dtype=torch.int64).unsqueeze(-1)
            probs=[]
            for i in range(env_info["n_agents"]):
                q,h=mac.agent(torch.cat([obs[i],lastAction[i],torch.tensor([i])]),mac.hs[i])
                actionMask=torch.tensor(sc_env.get_avail_agent_actions(i))
                action,prob=mac.agent.chooseAction(q,actionMask,args.epsilon)
                probs.append(prob)
                mac.hs[i]=h.detach()
                actions[i]=action
            reward, terminated, _ = sc_env.step(actions)
            t+=1
            epochTrainer.criticTrain(env_info["n_agents"],mac,obs,state,actions,lastAction,reward,args.gamma)
            epochTrainer.actorTrain(mac,env_info["n_agents"],actions,probs,state,obs,lastAction)
            lastAction=actions
        sc_env.close()




    
