from utils import oneHotTransform, onehot
from pscan import pscan
from smac.env import StarCraft2Env
from buffer import buffer
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
    buf=buffer(args.batchSize,args)
    epochTrainer=trainer(mac,args)
    epoch_i=0
    while epoch_i<args.epoch:
        with torch.no_grad():
            for episode_i in range(args.epochEpisodes):
                t=0
                ep_reward=0
                sc_env.reset()
                terminated = False
                mac.initHidden()
                episode=buffer(1,args)
                while not terminated: 
                    episode.data.obs[0,t] = torch.tensor(np.array(sc_env.get_obs()),device=device)
                    episode.data.state[0,t]=torch.tensor(sc_env.get_state(),device=device)
                    episode.data.availableActions[0,t]=torch.tensor(np.array(sc_env.get_avail_actions()),device=device)
                    actions=mac.chooseActions(episode,t,args.epsilon)   
                    reward, terminated, info = sc_env.step(actions)
                    episode.data.actions[0,t]=actions
                    episode.data.actionsOnehot[0,t]=oneHotTransform(actions,args.actionNum)
                    episode.data.rewards[0,t]=reward
                    t+=1
                    ep_reward+=reward
                    envTerminated=False
                    if terminated and not info.get("episode_limit", False):
                        envTerminated=True
                    episode.data.terminated[0,t]=envTerminated
                    episode.data.mask[0,t]=1
                buf.addEpisode(episode)
                print("episode: {}, steps: {}, total reward: {}".format(epoch_i,t,ep_reward))
        epoch_i+=args.epochEpisodes
        train=buf.canSample(args.sampleSize)
        if train:
            sampledData=buf.sample(args.sampleSize)
            epochTrainer.train(sampledData)


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





    
