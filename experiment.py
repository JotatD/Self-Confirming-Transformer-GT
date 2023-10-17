import gym
import numpy as np
import torch
import torch.nn.functional as F

import argparse
import os
import pickle
import random
import sys

from multiagent_particle_envs.make_env import make_env
from multiagent_particle_envs.multiagent.environment import BatchMultiAgentEnv
import time
from decision_transformer.evaluation.evaluate_episodes import batch_evaluate_versus_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.decision_transformer_controller import DecisionTransformerController
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.models.prey_model import PretrainedPrey
from datetime import datetime
import json

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


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
    adv_agents = 3
    #JUST STRINGS
    env_name, dataset = variant['env'], variant['dataset'] #FOR NAMING
    model_type = variant['model_type'] #FOR IF ELSE IN BEHAVIOR CLONING

    K = variant['K']    
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    #GET CONFIGS, HERE WE GET THE ENVIRONMENT ENV
    if env_name == "simple_tag":
        env_batch = BatchMultiAgentEnv([make_env(env_name) for i in range(num_eval_episodes)])
        cur_agents = env_batch.advs_agents_num
        opo_agents = env_batch.good_agents_num
        ep_len = 25
        obs_space = env_batch.observation_space
        act_space = env_batch.action_space
        state_dim = obs_space[0].shape[0]
        act_dim = act_space[0].shape[0]
        env_targets = range(0, 500, 20)
        scale = variant['scale']
        slicing = [[0, cur_agents], [cur_agents, opo_agents+cur_agents]]
        max_ep_len = 26
        is_discrete=False
        oponent = PretrainedPrey('decision_transformer/models/pretrained_tag.pt', device=variant["device"], input_dim=obs_space[-1].shape[0], output_dim=act_dim, hidden=64, discrete=is_discrete)
        dataset_path = variant['path']+"/"+variant['type']+"/"+variant['dataset']
        
    elif env_name == "simple_world":
        env_batch = BatchMultiAgentEnv([make_env(env_name) for i in range(num_eval_episodes)])
        cur_agents = env_batch.advs_agents_num
        opo_agents = env_batch.good_agents_num
        ep_len = 25
        obs_space = env_batch.observation_space
        act_space = env_batch.action_space
        state_dim = obs_space[0].shape[0]
        act_dim = act_space[0]._shape[0]
        slicing = [[0, cur_agents], [cur_agents, opo_agents+cur_agents]]
        env_targets = range(0, 500, 20)
        scale = variant['scale']
        max_ep_len = 26
        is_discrete=False
        oponent = PretrainedPrey('decision_transformer/models/pretrained_world.pt', device=variant["device"], input_dim=obs_space[-1].shape[0], output_dim=act_dim, hidden=64, discrete=False)
        dataset_path = variant['path']+"/"+variant['type']+"/"+variant['dataset']
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
    
    observations = []
    next_observations = []
    cur_acts = []
    opo_acts = []
    # load dataset
    for i in range(cur_agents):
        obs = np.load(dataset_path+'/'+'obs_{}.npy'.format(i)).reshape(-1, state_dim)  
        observations.append(obs)
        next_obs = np.load(dataset_path+'/'+'next_obs_{}.npy'.format(i)).reshape(-1, state_dim)
        next_observations.append(next_obs)
        cact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        cur_acts.append(cact)
        
    for i in range(cur_agents, cur_agents + opo_agents):
        oact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        opo_acts.append(oact)
    
    observations = np.stack(observations, axis = -2).reshape(-1, ep_len, cur_agents, state_dim)
    next_observations = np.stack(next_observations, axis = -2).reshape(-1, ep_len, cur_agents, state_dim)
    cur_acts = np.stack(cur_acts, axis = -2).reshape(-1, ep_len, cur_agents, act_dim)
    opo_acts = np.stack(opo_acts, axis = -2).reshape(-1, ep_len, opo_agents, act_dim)
    rewards = np.load(dataset_path+'/'+'rews_0.npy').reshape(-1, 25)

    # ######################DATASET LOADING####################
    observations = []
    next_observations = []
    cur_acts = []
    opo_acts = []
    for i in range(cur_agents):
        obs = np.load(dataset_path+'/'+'obs_{}.npy'.format(i)).reshape(-1, state_dim)  
        observations.append(obs)
        next_obs = np.load(dataset_path+'/'+'next_obs_{}.npy'.format(i)).reshape(-1, state_dim)
        next_observations.append(next_obs)
        cact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        cur_acts.append(cact)
        
    for i in range(cur_agents, cur_agents + opo_agents):
        oact =  np.load(dataset_path+'/'+'acs_{}.npy'.format(i)).reshape(-1, act_dim)
        opo_acts.append(oact)
    
    observations = np.stack(observations, axis = -2).reshape(-1, ep_len, cur_agents, state_dim)
    next_observations = np.stack(next_observations, axis = -2).reshape(-1, ep_len, cur_agents, state_dim)
    cur_acts = np.stack(cur_acts, axis = -2).reshape(-1, ep_len, cur_agents, act_dim)
    opo_acts = np.stack(opo_acts, axis = -2).reshape(-1, ep_len, opo_agents, act_dim)
    rewards = np.load(dataset_path+'/'+'rews_0.npy').reshape(-1, 25)
    
    trajectories = []
    for i in range(rewards.shape[0]):  
        trajectory = {
            'rewards': rewards[i],
            'observations': observations[i],
            'next_observations': next_observations[i],
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
    # used for input normalization
    state_mean, state_std = np.mean(observations, axis=(0,1,3)).reshape(1, 1, -1, 1), np.std(observations, axis=(0,1,3)).reshape(1, 1, -1, 1) + 1e-6
    if variant['noise']:
        noise = np.random.normal(scale=0.1*np.std(observations), size=observations.shape)
        observations = observations + noise
        next_observations = next_observations + noise

    num_trajectories = observations.shape[0]
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

        s, ns, opo_a, cur_a, r, rtg, timesteps, mask = [], [], [], [], [], [], [], []

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
            si = random.randint(0, trajectory['rewards'].shape[0]-K)

            # GET STATE ACTION AND REWARD SEQUENCES
            s.append(trajectory['observations'][si:si + max_len].reshape(1, -1, adv_agents, state_dim))
            ns.append(trajectory['next_observations'][si:si + max_len].reshape(1, -1, adv_agents, state_dim))
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
            s[-1] = np.concatenate([np.zeros((1, max_len-tlen, adv_agents, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            ns[-1] = np.concatenate([np.zeros((1, max_len-tlen, adv_agents, state_dim)), ns[-1]], axis=1)
            ns[-1] = (ns[-1] - state_mean) / state_std
            opo_a[-1] = np.concatenate([np.ones((1, max_len - tlen, 1, temp_act_dim)) * -10., opo_a[-1]], axis=1)
            cur_a[-1] = np.concatenate([np.ones((1, max_len - tlen, adv_agents, temp_act_dim)) * -10., cur_a[-1]], axis=1)
            # PAD WITH 0 THE REWARDS
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            
            # PAD WITH THE THE LAST THE RTG
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale

            # THE LAST TIMESTEPS ARE PADDED WITH 0
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            
            # MASK OF 1 MASKED AND 0 NOT MASKED
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        # FLATTEN PRESERVING THE LAST DIMENSION OR REMOVING THE SINGLETON LAST
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        ns = torch.from_numpy(np.concatenate(ns, axis=0)).to(dtype=torch.float32, device=device)
        opo_a = torch.from_numpy(np.concatenate(opo_a, axis=0)).to(dtype=act_datatype, device=device)
        cur_a = torch.from_numpy(np.concatenate(cur_a, axis=0)).to(dtype=act_datatype, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        return rtg, s, ns, opo_a, cur_a, timesteps, r, mask
    #######################SAMPLING########################

    #################EVALUATION METHOD####################
    def eval_episodes(target_rew):
        def fn(model):
            dt_c = DecisionTransformerController(
                model,
                target_rew, 
                state_dim,
                act_dim,
                cur_agents,
                opo_agents,
                n=num_eval_episodes,
                opo_final=variant['opo_final'],
                forward = variant['forward'],
                scale=scale,
                state_mean= torch.tensor(state_mean, device=variant['device']),
                state_std = torch.tensor(state_std, device=variant['device']),
                device=variant['device']
            )
            dt_c.reset_memory()
            with torch.no_grad():
                ret, dist = batch_evaluate_versus_episode_rtg(
                    env_batch,
                    oponent,
                    dt_c,
                    slicing=slicing,
                    max_ep_len=25,
                    is_forcing=True,
                    is_render=False
                )
            return {
                f'target_{target_rew}_standa': ret.std(),
                f'target_{target_rew}_return': ret.mean(),
                f'target_{target_rew}_distan': dist.mean(),
            }
        return fn
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
            is_discrete=False,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
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
    ws = 4*ws/np.sum(ws)
    def weighted_loss_fn(s_hat, oa_hat, ca_hat, r_hat, s, oa, ca, r):
        return ws[0]*torch.mean((s_hat-s)**2) + ws[1]*torch.mean((oa_hat - oa)**2) + ws[2]*torch.mean((ca_hat - ca)**2) + ws[3]*torch.mean((r_hat-r)**2)
    ###############TRAINER#######################
    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            # loss_fn=lambda s_hat, oa_hat, ca_hat, r_hat, s, oa, ca, r: torch.mean((oa_hat - oa)**2)+torch.mean((ca_hat - ca)**2),
            loss_fn=weighted_loss_fn,
            # WE PASS THE EVALUATION TO THE TRAINER, WITH MULTIPLE TARGET REWARDS
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            env_targets=env_targets,
            save=variant['save_model']
        )
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
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--opo_ope', type=str, default='normal')  # normal for prediction form obs, add for adding opo hidden state, cat for concat
    parser.add_argument('--opo_hid', type=str, default='transformer')  # transformer for trasformer hidden, mlp for hidden state based on the realized prediciton
    parser.add_argument('--forward', type=str, default='single')  # single for one forward, double for two forward
    parser.add_argument('--opo_final', type=str, default='zero') #zero for 0 padding, repeat for repeating the last action
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--eval', type=bool, default=0) 
    parser.add_argument('--ws', nargs='*', type=int, default=[1,1,1,1])
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--scale', type=int, default=100)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
