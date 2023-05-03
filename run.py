from utils import oneHotTransform, onehot
from pscan import pscan
from smac.env import StarCraft2Env
from buffer import buffer
import torch
import numpy as np
from trainer import trainer
import matplotlib.pyplot as plt
import time
import random
import sys
from rl_plotter.logger import Logger


print("cuda status: ",torch.cuda.is_available())
device = torch.device("cuda")
seed=random.randint(0,sys.maxsize)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def train(args):
    #plt
    if not args.debug:
        localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        with open("./results/pscan{}.txt".format(localtime), encoding="utf-8",mode="a") as file:  
            file.write("seed: {}\n".format(seed)) 
            file.write("mixer: {}\n".format(args.mixer)) 
            file.write("map: {}\n".format(args.map)) 

        logger = Logger(exp_name="coma+"+args.mixer, env_name=args.map,filename="r.csv")
        custom_logger=logger.new_custom_logger("wr.csv",fieldnames=["wr","episode"])

    sc_env = StarCraft2Env(map_name=args.map)#,reward_win=50
    env_info = sc_env.get_env_info()
    args.agentNum=env_info["n_agents"]
    args.observeDim=env_info["obs_shape"]
    args.actionNum=env_info["n_actions"]
    args.stateDim=env_info["state_shape"]
    args.maxSteps=env_info["episode_limit"]
    mac = pscan(args)
    buf=buffer(args.batchSize,args,"cpu")
    epochTrainer=trainer(mac,args)
    epoch_i=0
    t_env=0
    while epoch_i<args.epoch:
        with torch.no_grad():
            for episode_i in range(args.epochEpisodes):
                t=0
                ep_reward=0
                sc_env.reset()
                terminated = False
                mac.initHidden(1)
                episode=buffer(1,args)
                episode.data.obs[0,0] = torch.tensor(np.array(sc_env.get_obs()),device=device)
                episode.data.states[0,0]=torch.tensor(sc_env.get_state(),device=device)
                episode.data.availableActions[0,0]=torch.tensor(np.array(sc_env.get_avail_actions()),device=device)
                episode.data.mask[0,0]=1
                while not terminated: 
                    actions,probs=mac.chooseActions(episode,t,t_env)
                    reward, terminated, info = sc_env.step(actions.reshape(-1))
                    episode.data.actions[0,t]=actions
                    episode.data.actionsOnehot[0,t]=oneHotTransform(actions.reshape(-1,1),args.actionNum)
                    episode.data.rewards[0,t]=reward
                    episode.data.probs[0,t]=probs
                    ep_reward+=reward
                    envTerminated=False
                    if terminated and not info.get("episode_limit", False):
                        envTerminated=True
                    episode.data.terminated[0,t]=envTerminated
                    t+=1
                    episode.data.mask[0,t]=1
                    episode.data.obs[0,t] = torch.tensor(np.array(sc_env.get_obs()),device=device)
                    episode.data.states[0,t]=torch.tensor(sc_env.get_state(),device=device)
                    episode.data.availableActions[0,t]=torch.tensor(np.array(sc_env.get_avail_actions()),device=device)
                buf.addEpisode(episode)
                del episode
                t_env+=t
                won=False
                if info.get("battle_won", False):
                    won=True
                
                print("episode: {}, steps: {}, total reward: {}, won: {}".format(epoch_i+episode_i,t,ep_reward,won))
        epoch_i+=args.epochEpisodes
        train=buf.canSample(args.sampleSize)
        if train:
            sampledData=buf.sample(args.sampleSize,"cuda")
            epochTrainer.train(sampledData)
            del sampledData

        #test
        if epoch_i % 128 == 0:
            winCnt=0
            testRewards=[]
            with torch.no_grad():
                for i in range(100):
                    t=0
                    ep_reward=0
                    sc_env.reset()
                    terminated = False
                    mac.initHidden(1)
                    episode=buffer(1,args)
                    episode.data.obs[0,0] = torch.tensor(np.array(sc_env.get_obs()),device=device)
                    episode.data.availableActions[0,0]=torch.tensor(np.array(sc_env.get_avail_actions()),device=device)
                    while not terminated: 
                        actions,probs=mac.chooseActions(episode,t,0,True)
                        reward, terminated, info = sc_env.step(actions.reshape(-1))
                        episode.data.actionsOnehot[0,t]=oneHotTransform(actions.reshape(-1,1),args.actionNum)
                        ep_reward+=reward
                        t+=1
                        episode.data.obs[0,t] = torch.tensor(np.array(sc_env.get_obs()),device=device)
                        episode.data.availableActions[0,t]=torch.tensor(np.array(sc_env.get_avail_actions()),device=device)
                    won=False
                    if info.get("battle_won", False):
                        winCnt+=1
                        won=True
                    testRewards.append(ep_reward)
                    print("test: {}, steps: {}, total reward: {}, won: {}".format(i,t,ep_reward,won))

            wr=winCnt/100
            if not args.debug:
                with open("./results/pscan{}.txt".format(localtime), encoding="utf-8",mode="a") as file:  
                    file.write("{}    {}\n".format(epoch_i,wr))
                logger.update(testRewards,epoch_i)
                custom_logger.update([wr,epoch_i],t_env)
    





    
