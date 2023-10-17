from tkinter.messagebox import NO
import numpy as np
import torch
from torch import softmax, multinomial
import time
import copy

def batch_evaluate_versus_episode_rtg(batch_env, oponent, controller, slicing, max_ep_len, is_forcing=False, is_render=False):
    n = batch_env.len
    prev_cur_actions = np.empty(n)
    prev_opo_actions = np.empty(n)
    rewards = np.zeros(n)
    total_rewards = np.zeros(n)
    distances = np.zeros((n,max_ep_len))
    states = batch_env.reset()
    cur_states = np.array([state[slicing[0][0]:slicing[0][1]] for state in states])
    opo_states = np.array([state[slicing[1][0]:slicing[1][1]] for state in states])
    opo_states = np.float32(opo_states)
    for i in range(max_ep_len):
        opo_preds, cur_actions = controller.get_action(prev_opo_actions, prev_cur_actions, rewards, cur_states)

        opo_actions = oponent.step(opo_states)
        cur_actions = cur_actions.detach().cpu().numpy()
        opo_actions = opo_actions.detach().cpu().numpy()
       
        if opo_preds!=None:
            opo_preds = opo_preds.detach().cpu().numpy()
            distances[:,i] = np.linalg.norm(opo_preds - opo_actions, axis=-1)[:,0]
        
        prev_opo_actions = opo_actions.copy() if is_forcing else opo_preds.copy()
        prev_cur_actions = cur_actions.copy()
        
        formated_actions = [format_action(opo_action, cur_action) for opo_action, cur_action in zip(opo_actions, cur_actions)]
        states, rewards, _, _ = batch_env.step(formated_actions)
        
        if is_render:
            
            batch_env.env_batch[0].render()
        rewards = np.array(rewards)[:, slicing[0][0]]
        total_rewards += rewards

        cur_states = np.array([state[slicing[0][0]:slicing[0][1]] for state in states])
        opo_states = np.array([state[slicing[1][0]:slicing[1][1]] for state in states])
        opo_states = np.float32(opo_states)

    return total_rewards, distances

def format_action(opoact, curact):
    action = []
    for i in range(curact.shape[0]):
        action.append(curact[i])

    for i in range(opoact.shape[0]):
        action.append(opoact[i])
    return action