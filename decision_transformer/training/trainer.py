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


        print("--------------TRAINING TIME-------------")
        train_start = time.time()
        self.model.train()
        for z in range(num_steps):
            train_loss = self.train_step(z)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()
                
            print("train step")

        logs['time/training'] = time.time() - train_start


        print("--------------EVALUATION TIME-------------")
        eval_start = time.time()
        self.model.eval()
        d = {}
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                d.setdefault(k, []).append(v)
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
            
        if print_logs:
            print('=' * 80 + '\n')
            print(f'Iteration {iter_num}\n')
            for k, v in logs.items():
                    log_line = f'{k}: {v}\n'
                    print(log_line, end='')
            
        return logs

