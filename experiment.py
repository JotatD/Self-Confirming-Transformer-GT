import gym
import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import random

from multiagent_particle_envs.make_env import make_env
from multiagent_particle_envs.multiagent.environment import BatchMultiAgentEnv
from decision_transformer.evaluation.evaluate_episodes import batch_evaluate_versus_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_transformer_controller import DecisionTransformerGTController
from decision_transformer.training.seq_trainer import SequenceTrainer
from datetime import datetime
import json
from agents.agents import TitForTat

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

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

def experiment(
        exp_prefix, #a strinb
        variant, #arguments
):
    if variant['eval']:
        variant['max_iters'] = 1
        variant['num_steps_per_iter'] = 1
        variant['num_eval_episodes'] = 100
    #ARGUMENTS
    device = variant.get('device', 'cuda')
    #JUST STRINGS
    env_name, dataset = variant['env'], variant['dataset'] #FOR NAMING
    model_type = variant['model_type'] #FOR IF ELSE IN BEHAVIOR CLONING

    K = variant['K']    
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    #GET CONFIGS, HERE WE GET THE ENVIRONMENT ENV
    if env_name == "iterated_prisoner":
        cur_agents = 1
        opo_agents = 1
        ep_len = 100
        state_dim = 0
        adv_agents = 1
        K = 100
        act_dim = 1
        slicing = [[0, 1], [1, 2]]
        env_targets = 3000
        scale = variant['scale']
        max_ep_len = 100
        is_discrete=True
        oponent = TitForTat()
        dataset_path = variant['dataset_path']
    else:
        raise NotImplementedError

    ######################DATASET LOADING####################

    # The structure of our dataset is like:
    # seed_0/1/2/3/4_data
    # |-- obs_0.npy     for agent 1
    # |-- obs_1.npy     for agent 2
    # |-- ......
    # also, the data is in shape (100000, dim) action dim or state dim
    # and we need to devide the data into 25 steps for each episode
    
    # ######################DATASET LOADING####################
    cur_acts = []
    opo_acts = []
    for i in range(cur_agents):
        cact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        cur_acts.append(cact)
        
    for i in range(cur_agents, cur_agents + opo_agents):
        oact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        opo_acts.append(oact)
    
    cur_acts = np.stack(cur_acts, axis = -2).reshape(-1, ep_len, cur_agents, act_dim)
    opo_acts = np.stack(opo_acts, axis = -2).reshape(-1, ep_len, opo_agents, act_dim)
    rewards = np.load(dataset_path+'/'+'rews_0.npy').reshape(-1, ep_len)
    
    trajectories = []
    for i in range(rewards.shape[0]):  
        trajectory = {
            'rewards': rewards[i],
            'cur_acts': cur_acts[i],
            'opo_acts': opo_acts[i],
        }
        trajectories.append(trajectory)
        
    ################### DATA IS SEPARATED####################
    ''' DATA IS COMPOSED OF TRAJECTORIES 
    SEPARATED EACH IN A DICTIONARY
    EACH TRAJECTORY HAS REWARDS, A LIST OF OBSERVATIONS, AND LIST OF ACTIONS
    '''

    #######################DATA PROCESSING##################
    num_trajectories = cur_acts.shape[0]
    #######################DATA PROCESSING##################
    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{num_trajectories} timesteps found')

    print('=' * 50)

    ################## SAMPLING #########################
    def get_batch(batch_size=256, epoch = 0, max_len=K):
        # GENERATE RANDOM INDICES FOR THE TRAJECTORIES
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
        )

        cur_a, opo_a, r, rtg, timesteps, mask = [], [], [], [], [], []

        if is_discrete:
            temp_act_dim = 1
            act_datatype = torch.long
        else:
            temp_act_dim = act_dim
            act_datatype = torch.float32
            
        for i in batch_inds:
            i = int(i)
            trajectory = trajectories[i]
            # GENERATE RANDOM INDICES FOR THE TIMESTEPS
            si = random.randint(0, trajectory['rewards'].shape[0]-1)

            # GET STATE ACTION AND REWARD SEQUENCES
            opo_a.append(trajectory['opo_acts'][si:si + max_len].reshape(1, -1, 1, act_dim))
            cur_a.append(trajectory['cur_acts'][si:si + max_len].reshape(1, -1, adv_agents, act_dim))
            r.append(trajectory['rewards'][si:si + max_len].reshape(1, -1, 1))
            
            tlen = opo_a[-1].shape[1]
            timesteps.append(np.arange(si, si + tlen).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1 
            
            # CALCULATE the return to go
            rtg.append(discount_cumsum(trajectory['rewards'][si:], gamma=1.)[:tlen + 1].reshape(1, -1, 1))
            
            if rtg[-1].shape[1] <= tlen:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            
            # PAD WITH 0s AND THEN NORMALIZE
            opo_a[-1] = np.concatenate([np.ones((1, max_len - tlen, 1, temp_act_dim)) * 2, opo_a[-1]], axis=1)
            cur_a[-1] = np.concatenate([np.ones((1, max_len - tlen, adv_agents, temp_act_dim)) * 2, cur_a[-1]], axis=1)
            # PAD WITH 0 THE REWARDS
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            
            # PAD WITH THE THE LAST THE RTG
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale

            # THE LAST TIMESTEPS ARE PADDED WITH 0
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            
            # MASK OF 1 MASKED AND 0 NOT MASKED
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        # FLATTEN PRESERVING THE LAST DIMENSION OR REMOVING THE SINGLETON LAST
        opo_a = torch.from_numpy(np.concatenate(opo_a, axis=0)).to(dtype=act_datatype, device=device)
        cur_a = torch.from_numpy(np.concatenate(cur_a, axis=0)).to(dtype=act_datatype, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return rtg, opo_a.to(dtype=torch.long), cur_a.to(dtype=torch.long), timesteps, r, mask
    #######################SAMPLING########################

    #################EVALUATION METHOD####################
    def evaluate(model, n_games=25):
        dt_c = DecisionTransformerGTController(
                model,
                3000,
                act_dim,
                cur_agents,
                opo_agents,
                n=1,
                forward = variant['forward'],
                scale=scale,
                device=variant['device']
            )
        total_rewards = np.zeros((n_games, 100))
        model_preds = np.zeros((n_games, 100))
        p2_actions = np.zeros((n_games, 100))
        env = gym.make('custom_envs.iterated_games.iterated_prisoner:IteratedPrisoner-v0')
        for i in range(n_games):
            dt_c.reset_memory()
            _, acts_p2, rews1, _, preds= game(env, dt_c, oponent, game_horizon=100)
            total_rewards[i] = rews1
            model_preds[i] = preds
            p2_actions[i] = acts_p2
            print(f"game {i} = {rews1.sum()}")            
        return {
            'rewards_against_TFT': np.sum(total_rewards),
            'accuracy': np.mean(model_preds == p2_actions),
        }
    #################EVALUATION METHOD####################

    ################MODEL#########################
    if model_type == 'dt':
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
    else:
        raise NotImplementedError
    
    model = model.to(device=device)
    
    ################MODEL#########################

    #################OPTIMIZER AND SCHEDULER####################
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    #################OPTIMIZER AND SCHEDULER####################
    ws = np.array(variant['ws'])
    from torch.nn.functional import cross_entropy
    
    
    
    def weighted_loss_fn(oa_hat, ca_hat, r_hat, oa, ca, r):
        return  ws[0]*cross_entropy(oa_hat, oa.squeeze(1)) +\
                ws[1]*cross_entropy(ca_hat, ca.squeeze(1)) +\
                ws[2]*torch.mean((r_hat-r)**2)
                
                
                
    ###############TRAINER#######################
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=weighted_loss_fn,
            # WE PASS THE EVALUATION TO THE TRAINER, WITH MULTIPLE TARGET REWARDS
            env_targets=env_targets,
            save=variant['save_model'],
            eval_fns=[evaluate])
    ###############TRAINER#######################

    if variant['load'] != '':
        print("Model loading!")
        model.load_state_dict(torch.load(variant['load']))
        print("Model loaded!")

    time = datetime.now().strftime('%m-%d %H:%M:%S')

    seed = str(torch.seed())[:5]
    print("Experiment seed is = ", {seed})
    saving_path = f"saved/{env_name} {variant['type']} {time}"
    print(saving_path)
    if not variant['eval']: 
        if variant['save_model']:
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            with open(saving_path+"/config.json", 'w') as json_file:
                json.dump(variant, json_file, indent=4)

    #####FINALLY RUN#############
    for iter in range(variant['max_iters']):
        outputs= trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, saving_path=saving_path)
        if not variant['eval']:
            file_name = f'/{env_name}_iter_{iter}.model'

            if variant['save_model']:
                torch.save(trainer.model.state_dict(), saving_path+file_name)
    #########FINALLY RUN#############



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='simple_tag')
    parser.add_argument('--path', type=str)
    parser.add_argument('--type', type=str, default='expert')
    parser.add_argument('--dataset', type=str, default='seed_0_data')  # medium, medium-expert, expert
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10_000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=15)
    parser.add_argument('--num_steps_per_iter', type=int, default=10_000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--opo_ope', type=str, default='normal')  # normal for prediction form obs, add for adding opo hidden state, cat for concat
    parser.add_argument('--opo_hid', type=str, default='transformer')  # transformer for trasformer hidden, mlp for hidden state based on the realized prediciton
    parser.add_argument('--forward', type=str, default='single')  # single for one forward, double for two forward
    parser.add_argument('--opo_final', type=str, default='zero') #zero for 0 padding, repeat for repeating the last action
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--eval', type=bool, default=0) 
    parser.add_argument('--ws', nargs='*', type=int, default=[1,1,1])
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default='pd_dataset')
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
