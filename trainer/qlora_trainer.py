import os
import json
import copy
from collections import defaultdict
import argparse
import itertools
from tqdm import tqdm
import logging
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class QuantBaseTrainer:

    def __init__(self,
                 args: argparse.Namespace,
                 model,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR,
                 log: logging.Logger,
                 device: torch.device,
                ):

        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.train_sampler = iter(self.train_dataloader) if self.train_dataloader is not None else None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log = log
        self.device = device

    def eval(self):
        self.model.eval()
        predict = []
        targets = []

        avg_acc = 0.
        nstep = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.eval_dataloader)):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                outputs = self.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
                pred = torch.argmax(outputs['logits'], dim=1)
                avg_acc += torch.sum(pred == batch['labels']).item()
                nstep += len(batch['input_ids'])
        return avg_acc / nstep

    
    def __load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.log.info(f'Model loaded from {model_path}')
        
    def __save_model(self, step):
        model_path = os.path.join(self.args.save_dir, f'{self.args.model_type.split("/")[-1]}_{self.args.train_tasks.split("/")[-1]}_{self.args.method}_{step}.pth')
        torch.save(self.model.state_dict(), model_path)
        self.log.info(f'Model saved at {model_path}')

    def train(self):

        self.log.info('Training Start...')
        total_step = 0
        for e in range(self.args.num_epochs):
            self.model.train()
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param)
            # exit()
                # if 'holy' in name or 'score' in name or 'lora' in name:
                #     param.requires_grad = True
                # else:
                #     param.requires_grad = False
            for batch in tqdm(self.train_dataloader):
                # batch to tensor
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     labels=batch['labels'])
                loss = outputs['loss']
                # print(outputs.keys())
                # exit()

                # lm loss over generated tokens
                loss.backward()

                # print(self.model.model.layers[0].self_attn.q_proj.weight.grad)
                # exit()

                self.optimizer.step()
                self.scheduler.step()
                total_step += 1

                # eval and log
                if total_step % self.args.eval_interval == 0:
                    eval_acc = self.eval()
                    self.log.info(f'Epoch {e}, Step {total_step}, Eval Acc: {eval_acc}')
                    self.model.train()
                    for name, param in self.model.named_parameters():
                        if 'comp' in name and 'score' in name:
                            param.requires_grad = True

                if total_step % self.args.save_interval == 0:
                    self.__save_model(total_step)
            eval_acc = self.eval()
            self.log.info(f'Epoch {e}, Eval Acc: {eval_acc}')