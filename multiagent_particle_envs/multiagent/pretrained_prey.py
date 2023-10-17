import torch
import torch.nn as nn
import torch.nn.functional as F
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

class PretrainedPrey():
    def __init__(self, save_files, device='cpu', discrete=False, **kwargs):
        self.device = device
        self.discrete = discrete
        self.policy = MLP(**kwargs)
        save_dict = torch.load(save_files)
        self.policy.load_state_dict(save_dict['agent_params'][-1]['policy'])
        self.policy.to(self.device)
        self.policy.eval()
    
    def step(self, observation):
        observation = torch.tensor(observation).to(self.device, dtype=torch.float32).reshape(observation.shape[0], -1)
        actions = self.policy(observation)
        if self.discrete:
            actions = actions.argmax(dim=-1)
        else:
            actions = actions.clamp(-1, 1)
        return actions.detach().cpu().numpy()
