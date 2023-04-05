from utils import onehot
from pscan import pscan
from smac.env import StarCraft2Env
import torch
import numpy as np
from trainer import trainer
import matplotlib.pyplot as plt
import time


print("cuda status: ",torch.cuda.is_available())
device = torch.device("cuda")

def train(args):
    #plt
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) 
    x=[]
    y=[]

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
        ep_reward=0
        sc_env.reset()
        epochTrainer.initLast()
        terminated = False
        mac.initHidden()
        lastAction=torch.zeros(env_info["n_agents"],dtype=torch.int64,device=device).unsqueeze(-1)
        actions=torch.zeros(env_info["n_agents"],dtype=torch.int64,device=device).unsqueeze(-1)
        while not terminated: 
            obs = torch.tensor(np.array(sc_env.get_obs()),device=device)
            state=torch.tensor(sc_env.get_state(),device=device)
            probs=[]
            for i in range(env_info["n_agents"]):
                q,h=mac.agent(torch.cat([obs[i],onehot(lastAction[i],args.actionNum),onehot(i,args.agentNum)]),mac.hs[i])
                actionMask=torch.tensor(sc_env.get_avail_agent_actions(i),device=device)
                action,prob=mac.agent.chooseAction(q,actionMask,args.epsilon)
                probs.append(prob)
                mac.hs[i]=h.detach()
                actions[i]=action
            reward, terminated, info = sc_env.step(actions)
            t+=1
            ep_reward+=reward
            epochTrainer.criticTrain(env_info["n_agents"],mac,obs,state,actions,lastAction,reward,args.gamma)
            epochTrainer.actorTrain(mac,env_info["n_agents"],actions,probs,state,obs,lastAction)
            lastAction=actions
        #sc_env.close()
        print("episode: {}, steps: {}, total reward: {}".format(epoch_i,t,ep_reward))

        #eval every 100 episode
        if (epoch_i+1)%100==0:
            with torch.no_grad():
                wonCnt=0
                for evalEp in range(args.evalEp):
                    ep_reward=0
                    sc_env.reset()
                    terminated = False
                    mac.initHidden()
                    lastAction=torch.zeros(env_info["n_agents"],dtype=torch.int64,device=device).unsqueeze(-1)
                    actions=torch.zeros(env_info["n_agents"],dtype=torch.int64,device=device).unsqueeze(-1)
                    while not terminated:
                        obs = torch.tensor(np.array(sc_env.get_obs()),device=device)
                        state=torch.tensor(sc_env.get_state(),device=device)
                        for i in range(env_info["n_agents"]):
                            q,h=mac.agent(torch.cat([obs[i],onehot(lastAction[i],args.actionNum),onehot(i,args.agentNum)]),mac.hs[i])
                            actionMask=torch.tensor(sc_env.get_avail_agent_actions(i),device=device)
                            action,prob=mac.agent.chooseAction(q,actionMask,args.epsilon)
                            mac.hs[i]=h
                            actions[i]=action
                        reward, terminated, info = sc_env.step(actions)
                        ep_reward+=reward
                        if "battle_won" in info:
                            if info["battle_won"]:
                                wonCnt+=1
                                break
                        else:
                            info["battle_won"]=False
                        lastAction=actions
                    print("eval episode: {}, steps: {}, total reward: {}, won: {}".format(evalEp,t,ep_reward,info["battle_won"]))
                    #sc_env.close()
            wr=wonCnt/args.evalEp
            print("episode {}: win rate: {}".format(epoch_i+1,wr))
            x.append(epoch_i)
            y.append(wr)
            with open("./results/pscan{}.txt".format(localtime), encoding="utf-8",mode="a") as file:  
                file.write("{}    {}\n".format(epoch_i+1,wr)) 
    plt.plot(x,y)
    plt.savefig('./results/pscan{}.jpg'.format(localtime))





    
