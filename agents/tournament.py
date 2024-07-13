from agents import *
import gym
import pandas as pd
from functools import partial
import numpy as np
import os
from datetime import datetime
import pickle as pkl
import json

def game(env, player_1, player_2, game_horizon=100):
    env.reset()    
    actions_p1 = np.zeros(game_horizon)
    actions_p2 = np.zeros(game_horizon)
    rewards_p1 = np.zeros(game_horizon)
    rewards_p2 = np.zeros(game_horizon)
    
    for timestep in range(game_horizon):
        a1 = player_1.get_action()
        a2 = player_2.get_action()
        
        _, rewards, done, _ = env.step(a1, a2)
        
        player_1.update(a1, a2)
        player_2.update(a2, a1)
        
        actions_p1[timestep] = a1
        actions_p2[timestep] = a2
        rewards_p1[timestep] = rewards['player_1']
        rewards_p2[timestep] = rewards['player_2']
        
        if done:
            break
    
    return actions_p1, actions_p2, rewards_p1, rewards_p2
        

def tournament(players, env, n_games=100, game_horizon=100):
    names = list(players.keys())
    n_players = len(names)
    
    player_1_history = np.zeros((n_players, n_players, n_games, game_horizon))
    player_2_history = np.zeros((n_players, n_players, n_games, game_horizon))
    player_1_rewards = np.zeros((n_players, n_players, n_games, game_horizon))
    player_2_rewards = np.zeros((n_players, n_players, n_games, game_horizon))
    
    results = pd.DataFrame(0, index=names, columns=names)
    
    env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')
    for player_1_name in names:
        for player_2_name in names:
            p1_idx, p2_idx = names.index(player_1_name), names.index(player_2_name)
            print(f'Playing {player_1_name} vs {player_2_name}')
            for game_num in range(n_games):
                player_1 = players[player_1_name]()
                player_2 = players[player_2_name]()
                
                actions_p1, actions_p2, rewards_p1, rewards_p2 = game(env, player_1, player_2, game_horizon)
                player_1_history[p1_idx, p2_idx, game_num] = actions_p1
                player_2_history[p1_idx, p2_idx, game_num] = actions_p2
                player_1_rewards[p1_idx, p2_idx, game_num] = rewards_p1
                player_2_rewards[p1_idx, p2_idx, game_num] = rewards_p2
            results.loc[player_1_name, player_2_name] = np.sum(player_1_rewards[p1_idx, p2_idx])
    
    return results, player_1_history, player_2_history, player_1_rewards, player_2_rewards
            
            
    
if __name__ == '__main__':
    players = {
        'all_d': AllD,
        'all_c': AllC,
        'tit_for_tat': TitForTat,
        'test_agent': TestAgent,
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
    breakpoint()
    config = {
        'players': list(players.keys()),
        'env': 'custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0',
        'n_games': 1,
        'game_horizon': 1000
    }
    env = gym.make(config['env'])

    
    # create a folder with the current date as name
    export_base = 'pd_dataset/'
    export_folder = os.path.join(export_base, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(export_folder, exist_ok=True)
    results, player_1_history, player_2_history, player_1_rewards, player_2_rewards = tournament(players=players, env=env, n_games=config['n_games'], game_horizon=config['game_horizon'])
    #save results
    np.save(os.path.join(export_folder, 'player_1_history.npy'), player_1_history)
    np.save(os.path.join(export_folder, 'player_2_history.npy'), player_2_history)
    np.save(os.path.join(export_folder, 'player_1_rewards.npy'), player_1_rewards)
    np.save(os.path.join(export_folder, 'player_2_rewards.npy'), player_2_rewards)
    with open(os.path.join(export_folder, 'results.pkl'), 'wb') as f:
        pkl.dump(results, f)
    with open(os.path.join(export_folder, 'config.json'), 'w') as f:
        json.dump(config, f, separators=(',', ':'), indent=4)