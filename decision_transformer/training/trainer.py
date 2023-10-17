from collections import defaultdict
import numpy as np
import torch

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, env_targets=None, save=True):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.env_targets = env_targets
        self.start_time = time.time()
        self.save = save

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, saving_path=""):

        train_losses = []
        logs = dict()

        train_start = time.time()

        print("--------------TRAINING TIME-------------")
        self.model.train()
        for z in range(num_steps):
            train_loss = self.train_step(z)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
                
            # print("train step")

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        print("--------------EVALUATION TIME-------------")
        self.model.eval()
        d = {}
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                d.setdefault(k[-6:], []).append(v)
                logs[f'evaluation/{k}'] = v

        corr = np.corrcoef(np.array([self.env_targets, d['return'], d['distan']]))
        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if self.save:
            with open(saving_path+"/cout.txt", "a") as cout_file:  # "a" opens the file in append mode
                cout_file.write('=' * 80 + '\n')
                cout_file.write(f'Iteration {iter_num}\n')
                for k, v in logs.items():
                    log_line = f'{k}: {v}\n'
                    cout_file.write(log_line)

                cout_file.write(str(corr) + '\n')
                cout_file.write(str(d) + '\n')
                cout_file.write('last 50 return max ' + str(np.array(d['return'])[-50:].max()) + '\n')
                cout_file.write('last 50 return avg ' + str(np.array(d['return'])[-50:].mean()) + '\n')
                cout_file.write('last 50 distan avg ' + str(np.array(d['distan'])[-50:].mean()) + '\n')
        if print_logs:
            print('=' * 80 + '\n')
            print(f'Iteration {iter_num}\n')
            for k, v in logs.items():
                    log_line = f'{k}: {v}\n'
                    print(log_line, end='')

            print('last 50 return max ' + str(np.array(d['return'])[-50:].max()) + '\n')
            print('last 50 return avg ' + str(np.array(d['return'])[-50:].mean()) + '\n')
            print('last 50 distan avg ' + str(np.array(d['distan'])[-50:].mean()) + '\n')
            
        return logs

    def train_step(self, epoch):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size, epoch)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds,  = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
