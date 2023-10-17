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


#THIS MODELS ASSUMES TWO AGENTS, SAME ACTION SPACE
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

        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        
        self.is_discrete = is_discrete
        if self.is_discrete:
            self.embed_curact = nn.Embedding(self.act_dim+1, self.hidden_size)
            self.embed_opoact = nn.Embedding(self.act_dim+1, self.hidden_size)
        else:   
            self.embed_curact = torch.nn.Linear(self.act_dim, self.hidden_size)
            self.embed_opoact = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # some layers which are going to make use of opoact
        self.opo_type = opo_type
        self.opo_hid = opo_hid
        if self.opo_type == "cat":
            self.opo_obs_cat = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
        if self.opo_hid == "mlp":
            self.opo_act_lin = torch.nn.Linear(self.act_dim, self.hidden_size)
        assert not (self.opo_type == "normal" and self.opo_hid == "mlp")
            
        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)

        self.predict_opoact = nn.Sequential(
            *([nn.Linear(3*self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_curact = nn.ModuleList()

        for i in range(3):
            actor = nn.Sequential(
                *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )   
            self.predict_curact.append(actor)

        self.predict_return = torch.nn.Linear(6 * self.hidden_size, 1)
        

    def forward(self, returns_to_go, states, opoacts, curacts, timesteps, rewards=0, attention_mask=None):
        # return to go      states              opoacts            curacts
        # [bs, l, 1, eb]   [bs, l, ag, eb]     [bs, l, 1, eb]    [bs, l, ag, eb]

        batch_size, seq_length, opo_agents, cur_agents= states.shape[0], states.shape[1], opoacts.shape[2], curacts.shape[2]
        total =  1 + cur_agents + opo_agents + cur_agents

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        
        state_embeddings = self.embed_state(states)
        curact_embeddings = self.embed_curact(curacts)
        opoact_embeddings = self.embed_opoact(opoacts)
        returns_embeddings = self.embed_return(returns_to_go).reshape(batch_size, -1, 1, self.hidden_size)
        time_embeddings = self.embed_timestep(timesteps).reshape(batch_size, -1, 1, self.hidden_size)
        
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        curact_embeddings = curact_embeddings + time_embeddings
        opoact_embeddings = opoact_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.cat(
            (returns_embeddings, state_embeddings, opoact_embeddings, curact_embeddings
        ), dim=2).reshape(batch_size, total*seq_length, self.hidden_size)
        
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
        # returns (0), states (1-3), opoacts(4), curacts(5-7); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, total, self.hidden_size)

        action_hidden_states = x[:,:,1 + cur_agents + opo_agents:1 + cur_agents + opo_agents + cur_agents]
        observation_hidden_states = x[:,:,1:1+cur_agents]
        oponent_hidden_states = x[:,:,1+cur_agents:1+cur_agents+opo_agents]
        
        # predict return based on action and state
        return_preds = self.predict_return(torch.cat((action_hidden_states, observation_hidden_states), dim=-1).reshape(batch_size, seq_length, -1))
        state_preds  = self.predict_state(action_hidden_states) # predict next state given state, opoact, curact
        opoact_preds = self.predict_opoact(observation_hidden_states.reshape(batch_size, seq_length, -1)) # predict opponent's actions from state
        
        if self.opo_type == 'normal':
            curact_inp = observation_hidden_states
        else:
            if self.opo_hid == "transformer":
                to_stack = oponent_hidden_states
            elif self.opo_hid == "mlp":
                to_stack = self.opo_act_lin(opoact_preds).unsqueeze(-2)
            else:
                raise NotImplementedError
            
            opoact_hidden_stack = torch.cat((to_stack, to_stack, to_stack), dim=-2)
            
            if self.opo_type == 'add':
                curact_inp = (observation_hidden_states + opoact_hidden_stack)/2
            elif self.opo_type == 'cat':
                pre_curact_inp = torch.cat((observation_hidden_states, opoact_hidden_stack), dim=-1)
                curact_inp = self.opo_obs_cat(pre_curact_inp)
            else:
                raise NotImplementedError
        opoact_preds = opoact_preds[:, :, None, :]
        
        curact_preds = torch.empty(batch_size, seq_length, 0, 2).to(states.device)
        for n in range(len(self.predict_curact)):
            action = self.predict_curact[n](curact_inp[:,:,n]).reshape(batch_size, seq_length, 1, -1)
            curact_preds = torch.cat([curact_preds, action], dim=2)
        # curact_preds = self.predict_curact(curact_inp) # predict current player's actions from state
        return return_preds, state_preds, opoact_preds, curact_preds

    def get_action(self, returns_to_go, states, opoacts, curacts, timesteps, rewards=0,  **kwargs):
        # we don't care about the past rewards in this model
        
        # target_returns        states                      opo_acts                  cur_acts                   timesteps
        # [batch, l, 1]         [batch, l, 3, state_dim]   [batch, l, 1, act_dim]     [batch, l, 3, act_dim]     [batch, 1]

        if self.max_length is not None:
            returns_to_go = returns_to_go[:,-self.max_length:]
            states = states[:,-self.max_length:]
            opoacts = opoacts[:, -self.max_length:]
            curacts = curacts[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            tlen =  self.max_length-states.shape[1]
            # pad all tokens to sequence length
            attention_mask = torch.cat([
                torch.zeros([states.shape[0], tlen]),
                torch.ones([states.shape[0], states.shape[1]])], dim=1)
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], states.shape[2], states.shape[3]), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            
            opoacts = torch.cat(
                [torch.zeros((opoacts.shape[0], self.max_length-opoacts.shape[1], opoacts.shape[2], opoacts.shape[3]), device=states.device), opoacts],
                dim=1).to(dtype=torch.float32)
            
            curacts = torch.cat(
                [torch.zeros((curacts.shape[0], self.max_length-curacts.shape[1], curacts.shape[2], opoacts.shape[3]), device=curacts.device), curacts],
                dim=1).to(dtype=torch.float32)
            
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        # return to go      states              opoacts            curacts
        # [bs, l, 1, eb]   [bs, l, ag, eb]     [bs, l, 1, eb]    [bs, l, ag, eb]
        else:
            attention_mask = None
        _, _, opoact_preds, curact_preds= self.forward(
            returns_to_go, states, opoacts, curacts, timesteps, attention_mask=attention_mask,  **kwargs)
        return opoact_preds[:, -1], curact_preds[:, -1]
