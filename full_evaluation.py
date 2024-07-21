import numpy as np
import torch
import random
import gym
import argparse
import os

from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_transformer_controller import DecisionTransformerGTController
from agents.agents import *
from functools import partial
from datetime import datetime
import pickle as pkl
import json


  


def game(env, model, player_2, game_horizon=100):
    env.reset()    
    actions_p1 = np.zeros(game_horizon)
    actions_p2 = np.zeros(game_horizon)
    rewards_p1 = np.zeros(game_horizon)
    rewards_p2 = np.zeros(game_horizon)
    predictions_model= np.zeros(game_horizon)
    
    for timestep in range(game_horizon):
        a1, p1 = model.get_action_and_prediction()
        a2 = player_2.get_action()
        
        _, rewards, done, _ = env.step(a1, a2)
        
        model.update(a1, a2, rewards['player_1'])
        player_2.update(a2, a1)
        
        actions_p1[timestep] = a1
        actions_p2[timestep] = a2
        rewards_p1[timestep] = rewards['player_1']
        rewards_p2[timestep] = rewards['player_2']
        predictions_model[timestep] = p1
        
        if done:
            break
    
    return actions_p1, actions_p2, rewards_p1, rewards_p2, predictions_model

def model_round_roubbin(env, controller, players, n_games, horizon):
    total_a_p1 = np.zeros((len(players), n_games, horizon))
    total_a_p2 = np.zeros((len(players), n_games, horizon))
    total_r_p1 = np.zeros((len(players), n_games, horizon))
    total_r_p2 = np.zeros((len(players), n_games, horizon))
    total_preds = np.zeros((len(players), n_games, horizon))
    
    individual_stats = { p: {'rewards': 0, 'accuracy': 0} for p in players.keys()}
    total_stats = {'total_rewards': 0, 'total_accuracy': 0}
    for p, player in enumerate(players.keys()):
        for i in range(n_games):
            controller.reset_memory()
            opponent = players[player]()
            print(i, player)
            a_p1, a_p2, r_p1, r_p2, preds = game(env, controller, opponent, horizon)
            total_a_p1[p, i] = a_p1
            total_a_p2[p, i] = a_p2
            total_r_p1[p, i] = r_p1
            total_r_p2[p, i] = r_p2
            total_preds[p, i] = preds
            
    all_history = {
        'total_a_p1': total_a_p1,
        'total_a_p2': total_a_p2,
        'total_r_p1': total_r_p1,
        'total_r_p2': total_r_p2,
        'total_preds': total_preds
    }
    
    return all_history

def record_stats(history, players):
    total_a_p1, total_a_p2, total_r_p1, total_r_p2, total_preds = history.values()
    
    players_names = list(players.keys())
    
    individual_stats = {p: {'rewards': 0, 
                            'accuracy': 0} \
        for p in players.keys()}
    
    total_stats = {'total_rewards': 0,
                   'total_accuracy': 0,
                   'balance': 0}
    
    for p, player in enumerate(players_names):
        individual_stats[player]['rewards'] = total_r_p1[p].sum()
        individual_stats[player]['accuracy'] = (total_preds[p] == total_a_p2[p]).mean()
                
    total_stats['total_rewards'] = total_r_p1.sum()
    total_stats['total_accuracy'] = (total_preds == total_a_p2).mean() 
    total_stats['0_percentage'] = (total_a_p1 == 0).mean()
    
    return individual_stats, total_stats
            

def evaluation(variant):
    print(variant['target'])
    device = variant.get('device', 'cuda')
    env_name = variant['env']
    if env_name == "iterated_prisoner":
        cur_agents = 1
        opo_agents = 1
        state_dim = 0
        K = 100
        act_dim = 1
        scale = variant['scale']
        max_ep_len = 100
        is_discrete=True
        n_games = variant['n_games']
        target=variant['target']
        env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')
    else:
        raise NotImplementedError
    
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        hidden_size=variant['embed_dim'],
        opo_type=variant['opo_ope'],
        opo_hid=variant['opo_hid'],
        max_ep_len=max_ep_len,
        is_discrete=is_discrete,
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        act_space=2
    )
                
    if variant['load'] != '':
        print("Model loading!")
        model.load_state_dict(torch.load(variant['load']))
        print("Model loaded!")
        
    model = model.to(device=device)
    model.eval()
    
    dt_c = DecisionTransformerGTController(
        model,
        target,
        act_dim,
        cur_agents,
        opo_agents,
        n=1,
        forward = variant['forward'],
        scale=scale,
        device=variant['device']
    )
    
    players = {
        'all_d': AllD,
        'all_c': AllC,
        'tit_for_tat': TitForTat,
        'spiteful': Spiteful,
        'soft_majo': SoftMajo,
        'hard_majo': HardMajo,
        'per_ddc': PerDDC,
        'per_ccd': PerCCD,
        'mistrust': Mistrust,
        'per_cd': PerCD,
        'pavlov': Pavlov,
        'tf2t': TF2T,
        'hard_tft': HardTFT,
        'slow_tft': SlowTFT,
        'gradual': Gradual,
        'prober': Prober,
        'mem2': partial(Mem2, env_dict = env.payoff_dictionary)
    }
    
    all_history = model_round_roubbin(env, dt_c, players, n_games, K)
    individual_stats, total_stats = record_stats(all_history, players)
    
    date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    saving_folder = os.path.join(variant['saving_path'], date_string)
    
    os.makedirs(saving_folder, exist_ok=True)
    with open(os.path.join(saving_folder, 'all_history.pkl'), 'wb') as f:
        pkl.dump(all_history, f)
    with open(os.path.join(saving_folder, 'individual_stats.json'), 'w') as f:
        json.dump(individual_stats, f, indent=4)
    with open(os.path.join(saving_folder, 'total_stats.json'), 'w') as f:
        json.dump(total_stats, f, indent=4)
    with open(os.path.join(saving_folder, 'variant.json'), 'w') as f:
        json.dump(variant, f, indent=4)
        
    print('*' * 50)

    print('total_rewards = ', total_stats['total_rewards'])
    print('total_accuracy = ', total_stats['total_accuracy'])
    print('0_percentage = ', total_stats['0_percentage'])
    print('1_percentage = ', 1 - total_stats['0_percentage'])
    
    print('*' * 50)

    
    print(f'Indivudual rewards')
    for k, v in individual_stats.items():
        print(k, v['rewards'])
    
    print(f'\n\nIndivudual accuracy')
    for k, v in individual_stats.items():
        print(k, v['accuracy'])
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='iterated_prisoner')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10_000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--opo_ope', type=str, default='normal')  # normal for prediction form obs, add for adding opo hidden state, cat for concat
    parser.add_argument('--opo_hid', type=str, default='transformer')  # transformer for trasformer hidden, mlp for hidden state based on the realized prediciton
    parser.add_argument('--forward', type=str, default='single')  # single for one forward, double for two forward
    parser.add_argument('--opo_final', type=str, default='zero') #zero for 0 padding, repeat for repeating the last action
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument('--saving_path', type=str, default='evaluation_results/')
    parser.add_argument('--n_games', type=int, default=1)
    parser.add_argument('--target', type=int, default=300)

    
    args = parser.parse_args()
    
    seed_value = 2
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    evaluation(variant=vars(args))
