import gym
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, norm_in=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # create network layers
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(self.in_fn(x)))
        h = F.relu(self.fc2(h))
        out = self.fc3(h)
        return out

              
def to_continuous(actions):
    continuous = torch.zeros(actions.shape[0], 2).to(actions.device)
    for act, a in zip(actions, continuous):
        # process discrete action
        if act.item() == 1: a[0] = -1.0
        if act.item() == 2: a[0] = +1.0
        if act.item() == 3: a[1] = -1.0
        if act.item() == 4: a[1] = +1.0
    return continuous
    
class PretrainedPrey():
    def __init__(self, save_files, device='cpu',discrete=True, **kwargs):
        self.device = device
        self.discrete = discrete
        self.policy = MLP(**kwargs)
        save_dict = torch.load(save_files)
        self.policy.load_state_dict(save_dict['agent_params'][-1]['policy'])
        self.policy.to(self.device)
        self.policy.eval()
    
    def step(self, observation):
        observation = torch.tensor(observation).to(self.device, dtype=torch.float32).squeeze(1)
        actions = self.policy(observation)
        if self.discrete:
            actions = actions.argmax(dim=-1)
            actions = to_continuous(actions)
        else:
            actions = actions.clamp(-1, 1)
        return actions.unsqueeze(1)
    
class RandomPrey():
    def __init__(self, device):
        self.device = device
    def step(self, observations):
        observations = observations.squeeze(1)
        actions = torch.zeros(observations.shape[0], 2).to(self.device)
        for i, obs in enumerate(observations[:, 2:4]):
            while True:
                actions[i] = 2*torch.rand((1, 2), device=self.device) - 1
                if (
                    not (
                (abs(obs[0]) > 1 and np.sign(obs[0]) == np.sign(actions[i, 0])) or
                (abs(obs[1]) > 1 and np.sign(obs[1]) == np.sign(actions[i, 1]))
                        )
                    ):
                    break
        return actions.unsqueeze(1)

class MixedRandomPrey():
    def __init__(self, prey1, prob):
        self.prey1 = prey1
        self.prey2 = RandomPrey(prey1.device)
        self.prob = prob
    def step(self, observation):
        numb = random.random()
        if numb < self.prob:
            act = self.prey1.step(observation)
        else:
            act = self.prey2.step(observation)
        return act

class StillPrey():
    def __init__(self, device):
        self.device = device
    def step(self, observations):
        observations = observations.squeeze(1)
        actions = torch.zeros(observations.shape[0], 2).to(self.device)
        return actions.unsqueeze(1)             
                     


