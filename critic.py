import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):
    def __init__(self,intputSize,hiddenSize,outputSize):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(intputSize, hiddenSize)
        self.fc2=nn.Linear(hiddenSize, hiddenSize)
        self.fc3=nn.Linear(hiddenSize, outputSize)
    
    def forward(self, batch,n_agents,n_actions, t=None):
        inputs = self._build_inputs(batch, n_agents,n_actions,t=t)
        x = F.leaky_relu(self.fc1(inputs))
        x = F.leaky_relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch,n_agents,n_actions, t=None):
        bs = batch.batchSize
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch.data.states[:, ts].unsqueeze(2).repeat(1, 1, n_agents, 1))

        # observation
        inputs.append(batch.data.obs[:, ts])

        # actions (masked out by agent)
        actions = batch.data.actionsOnehot[:, ts].view(bs, max_t, 1, -1).repeat(1, 1, n_agents, 1)
        agent_mask = (1 - torch.eye(n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if t == 0:
            inputs.append(torch.zeros_like(batch.data.actionsOnehot[:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch.data.actionsOnehot[:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, n_agents, 1))
        else:
            last_actions = torch.cat([torch.zeros_like(batch.data.actionsOnehot[:, 0:1]), batch.data.actionsOnehot[:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, n_agents, 1)
            inputs.append(last_actions)

        inputs.append(torch.eye(n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = torch.cat([x.reshape(bs, max_t, n_agents, -1) for x in inputs], dim=-1)
        return inputs