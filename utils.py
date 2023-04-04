import torch

device = torch.device("cuda")

def hardUpdate(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def softUpdate(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def onehot(i,size):
    temp=torch.zeros(size,device=device)
    temp[int(i)]=1
    return temp