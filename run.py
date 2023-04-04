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
        lastAction=torch.zeros(env_info["n_agents"],dtype=torch.int64).cuda().unsqueeze(-1)
        actions=torch.zeros(env_info["n_agents"],dtype=torch.int64).cuda().unsqueeze(-1)
        while not terminated: 
            obs = torch.tensor(np.array(sc_env.get_obs())).cuda()
            state=torch.tensor(sc_env.get_state()).cuda()
            probs=[]
            for i in range(env_info["n_agents"]):
                q,h=mac.agent(torch.cat([obs[i],lastAction[i],torch.tensor([i]).cuda()]),mac.hs[i])
                actionMask=torch.tensor(sc_env.get_avail_agent_actions(i)).cuda()
                action,prob=mac.agent.chooseAction(q,actionMask,args.epsilon)
                probs.append(prob)
                mac.hs[i]=h.detach()
                actions[i]=action
            reward, terminated, _ = sc_env.step(actions)
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
                for evalEp in range(args.evalEp):
                    wonCnt=0
                    ep_reward=0
                    sc_env.reset()
                    terminated = False
                    mac.initHidden()
                    lastAction=torch.zeros(env_info["n_agents"],dtype=torch.int64).cuda().unsqueeze(-1)
                    actions=torch.zeros(env_info["n_agents"],dtype=torch.int64).cuda().unsqueeze(-1)
                    while not terminated:
                        obs = torch.tensor(np.array(sc_env.get_obs())).cuda()
                        state=torch.tensor(sc_env.get_state()).cuda()
                        for i in range(env_info["n_agents"]):
                            q,h=mac.agent(torch.cat([obs[i],lastAction[i],torch.tensor([i]).cuda()]),mac.hs[i])
                            actionMask=torch.tensor(sc_env.get_avail_agent_actions(i)).cuda()
                            action,prob=mac.agent.chooseAction(q,actionMask,args.epsilon)
                            mac.hs[i]=h
                            actions[i]=action
                        reward, terminated, info = sc_env.step(actions)
                        ep_reward+=reward
                        if info["battle_won"]:
                            wonCnt+=1
                            break
                        lastAction=actions
                    print("eval episode: {}, steps: {}, total reward: {}".format(evalEp,t,ep_reward))
                    #sc_env.close()
            wr=wonCnt/args.evalEp
            print("episode {}: win rate: {}".format(epoch_i+1,wr))
            x.append(epoch_i)
            y.append(wr)
            with open("./results/pscan{}.txt".format(localtime), encoding="utf-8",mode="a") as file:  
                file.write("{}    {}\n".format(epoch_i+1,wr)) 
    plt.plot(x,y)
    plt.savefig('./results/pscan{}.jpg'.format(localtime))





    
