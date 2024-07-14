from os import times
from pickletools import float8
from re import A
import numpy as np
import torch
import torch.nn as nn
import transformers
import time
from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            opo_type="normal",
            opo_hid="transformer",
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            is_discrete=True,
            action_space=2,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len+1, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        
        self.is_discrete = is_discrete
        self.act_space = action_space
        self.embed_curact = nn.Embedding(self.act_space, self.hidden_size)
        self.embed_opoact = nn.Embedding(self.act_space, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.opo_type = opo_type

        self.predict_opoact = nn.Linear(self.hidden_size, self.act_space)
        
        self.predict_curact = nn.Linear(self.hidden_size, self.act_space)

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        

    def forward(self, returns_to_go, opoacts, curacts, timesteps, rewards=0, attention_mask=None):
        # return to go      states              opoacts            curacts
        # [bs, l, 1, eb]   [bs, l, ag, eb]     [bs, l, 1, eb]    [bs, l, ag, eb]
        batch_size, seq_length, opo_agents, cur_agents= opoacts.shape[0], opoacts.shape[1], opoacts.shape[2], curacts.shape[2]
        total =  1  + opo_agents + cur_agents

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        
        curact_embeddings = self.embed_curact(curacts).reshape(batch_size, seq_length, cur_agents, self.hidden_size)
        # print(opoacts.reshape(-1))
        opoact_embeddings = self.embed_opoact(opoacts).reshape(batch_size, seq_length, opo_agents, self.hidden_size)
        returns_embeddings = self.embed_return(returns_to_go).reshape(batch_size, seq_length, 1, self.hidden_size)
        time_embeddings = self.embed_timestep(timesteps).reshape(batch_size, seq_length, 1, self.hidden_size)
        
        # time embeddings are treated similar to positional embeddings
        curact_embeddings = curact_embeddings + time_embeddings
        opoact_embeddings =  opoact_embeddings + time_embeddings
        returns_embeddings =  returns_embeddings + time_embeddings
        
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.cat(
            (returns_embeddings, opoact_embeddings, curact_embeddings), dim=2)
        stacked_inputs = stacked_inputs.reshape(batch_size, total*seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = attention_mask.unsqueeze(2).repeat(1, 1, total).reshape(batch_size, total*seq_length)
        # stacked_attention_mask = torch.stack(

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        
        x = transformer_outputs['last_hidden_state']
        # reshape x so that the second dimension corresponds to the original
        # returns (0), opoacts (1), curacts(2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, total, self.hidden_size)

        return_hidden_states = x[:, : , 0:1]
        opponent_hidden_states = x[ : , : , 1:2]
        cur_hidden_states = x[ : , : , 2:3]
        
        return_preds = self.predict_return(cur_hidden_states)
        opoact_preds  = self.predict_opoact(return_hidden_states) # predict next state given state, opoact, curact
        
        if self.opo_type == 'normal':
            curact_inp = opponent_hidden_states
        elif self.opo_type == 'no_opo':
            curact_inp = return_hidden_states
        else:
            raise NotImplementedError
        curact_preds = self.predict_curact(curact_inp) # predict current player's actions from state
        return return_preds, opoact_preds, curact_preds

    def get_action(self, returns_to_go, opoacts, curacts, timesteps, rewards=0,  **kwargs):
        if self.max_length is not None:
            returns_to_go = returns_to_go[:,-self.max_length:]
            opoacts = opoacts[:, -self.max_length:]
            curacts = curacts[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            tlen =  self.max_length-opoacts.shape[1]
            # pad all tokens to sequence length
            
            attention_mask = torch.cat([
                torch.zeros([opoacts.shape[0], tlen]),
                torch.ones([opoacts.shape[0], opoacts.shape[1]])], dim=1)
            attention_mask = attention_mask.to(dtype=torch.long, device=opoacts.device)
                        
            opoacts = torch.cat(
                [torch.zeros((opoacts.shape[0], self.max_length-opoacts.shape[1], opoacts.shape[2], opoacts.shape[3]), device=curacts.device), opoacts],
                dim=1).to(dtype=torch.int64)
            
            curacts = torch.cat(
                [torch.zeros((curacts.shape[0], self.max_length-curacts.shape[1], curacts.shape[2], opoacts.shape[3]), device=curacts.device), curacts],
                dim=1).to(dtype=torch.int64)
            
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)

        else:
            attention_mask = None
            
        _, opoact_preds, curact_preds= self.forward(returns_to_go, opoacts,
                        curacts, timesteps, attention_mask=attention_mask,  **kwargs)
        
        return opoact_preds[:, -1], curact_preds[:, -1]
