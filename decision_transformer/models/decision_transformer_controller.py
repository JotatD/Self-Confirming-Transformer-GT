import torch
import time

class DecisionTransformerController:
    def __init__(self,
                dt,
                rtg, 
                state_dim,
                act_dim,
                cur_agents,
                opo_agents,
                n=1,
                opo_final="zero",
                forward = 'single',
                scale=1000.,
                state_mean=0.,
                state_std=1.,
                device='cpu'):
        self.device = device
        self.dt = dt
        self.opo_agents = opo_agents
        self.cur_agents = cur_agents
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.scale = scale
        self.n = n # batch_env length, so we can do evaluation parallelly
        self.reset_memory()
        
        if opo_final == "zero":
            self.opo_acts_prepro = self.zero 
        elif opo_final == "repeat":
            self.opo_acts_prepro = self.repeat
        else:
            raise NotImplementedError
        
        if forward == 'single' or forward == 'double':
            self.forward = forward
        else:
            raise NotImplementedError
        
        self.configs = {'device':self.device, 'dtype':torch.float32}
        self.rtg = torch.tensor([rtg/self.scale], **self.configs)
        self.zero_cur_act = torch.zeros([self.n, 1, cur_agents, act_dim], **self.configs)
        self.zero_opo_act = torch.zeros([self.n, 1, opo_agents, act_dim],  **self.configs)
        self.unit_time =  torch.ones((self.n, 1), device=device, dtype=torch.long)
        
    def get_action(self, prev_opo_act, prev_cur_act, reward, cur_obs):
        self.total_rewards += torch.tensor(reward, **self.configs)
        cur_obs = torch.tensor(cur_obs, **self.configs).reshape([self.n, 1, self.cur_agents, self.state_dim])
        ret_to_go = (self.rtg - self.total_rewards/self.scale).reshape([self.n, 1, 1])
        
        self.opo_acts = torch.cat([self.opo_acts, self.zero_opo_act], dim=1)
        self.cur_acts = torch.cat([self.cur_acts, self.zero_cur_act], dim=1)
        self.rewards_to_go = torch.cat([self.rewards_to_go, ret_to_go], dim=1)
        self.observations = torch.cat([self.observations, cur_obs], dim=1)
        self.timesteps = torch.cat([self.timesteps,  self.unit_time * (self.t)], dim=1)
        
        if self.t != 0:
            prev_opo_act = torch.tensor(prev_opo_act, **self.configs).reshape([self.n, self.opo_agents, self.act_dim])
            prev_cur_act = torch.tensor(prev_cur_act, **self.configs).reshape([self.n, self.cur_agents, self.act_dim])
            self.opo_acts[:, -2] = prev_opo_act
            self.cur_acts[:, -2] = prev_cur_act
        
        self.opo_acts = self.opo_acts_prepro(self.opo_acts)
                
        opoact_pred, curact = self.forward_pass(self.rewards_to_go, self.observations, self.opo_acts, self.cur_acts, self.timesteps)
        
        self.t += 1
        return  opoact_pred, curact
        
    # set initial input of transformer    
    def reset_memory(self):
        self.rewards_to_go = torch.zeros([self.n, 0, 1], device = self.device, dtype=torch.float32)
        self.observations = torch.zeros([self.n, 0, self.cur_agents, self.state_dim], device = self.device, dtype=torch.float32)
        self.cur_acts = torch.zeros([self.n, 0, self.cur_agents, self.act_dim], device = self.device, dtype=torch.float32)
        self.opo_acts = torch.zeros([self.n, 0, self.opo_agents, self.act_dim], device = self.device, dtype=torch.float32)
        self.timesteps = torch.zeros([self.n, 0], device = self.device, dtype=torch.long)
        self.t = 0
        self.total_rewards = torch.zeros([self.n], device=self.device, dtype=torch.float32)
        self.dt.eval()
        self.dt.to(self.device)
        
    def zero(self, opo_acts):
        return opo_acts
    
    def repeat(self, opo_acts):
        opo_acts[:, -1] = opo_acts[:, -2] if self.t!= 0 else self.zero_opo_act.squeeze(1)
        return opo_acts
    
    def forward_pass(self, target_returns, states, opo_acts, cur_acts, timesteps):
        opoact_pred, curact_pred = self.dt.get_action(target_returns, 
                                    (states-self.state_mean)/self.state_std,
                                    opo_acts,
                                    cur_acts,
                                    timesteps
                                )
        if self.forward == "double":
            opo_acts[:, -1] = opoact_pred[:]
            _, curact_pred = self.dt.get_action(target_returns, 
                                    (states-self.state_mean)/self.state_std,
                                    opo_acts,
                                    cur_acts,
                                    timesteps
                                )
        
        return opoact_pred, curact_pred