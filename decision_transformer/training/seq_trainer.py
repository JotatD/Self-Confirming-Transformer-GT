import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, epoch):
        rtg, opoacts, curacts, timesteps, rewards, attention_mask = self.get_batch(self.batch_size, epoch)
        opoact_targets = torch.clone(opoacts)
        curact_targets = torch.clone(curacts)
        return_targets = torch.clone(rtg[:,1:])
        
        return_preds, opoact_preds, curact_preds = self.model.forward(
            rtg[:,:-1], opoacts, curacts, timesteps, rewards, attention_mask=attention_mask,
        )
        act_dim = curacts.shape[-1]
        act_space = curact_preds.shape[-1]
        cur_agents = curact_preds.shape[2]
        opo_agents = opoact_preds.shape[2]
        
        
        #TODO: Check correctness
        cur_attention_mask = torch.stack(
            [attention_mask for i in range(cur_agents)], dim=2)
        opo_attention_mask = torch.stack(
            [attention_mask for i in range(opo_agents)], dim=2)
                
        curact_preds = curact_preds.reshape(-1, cur_agents, act_space)[cur_attention_mask.reshape(-1, cur_agents) > 0]
        curact_targets = curact_targets.reshape(-1, cur_agents, act_dim)[cur_attention_mask.reshape(-1, cur_agents) > 0]
        
        opoact_preds = opoact_preds.reshape(-1, opo_agents, act_space)[opo_attention_mask.reshape(-1, opo_agents) > 0]
        opoact_targets = opoact_targets.reshape(-1, opo_agents, act_dim)[opo_attention_mask.reshape(-1, opo_agents) > 0]

        return_preds = return_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        return_targets = return_targets.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        loss = self.loss_fn(opoact_preds, curact_preds, return_preds, opoact_targets, curact_targets, return_targets,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        
        return loss.detach().cpu().item()
