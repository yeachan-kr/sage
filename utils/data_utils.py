
import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import random
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset
import copy 

from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def get_tokenizer(model_type: str, max_input_len: int = 512, padding_side="right"):
    # load decoder tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type, model_max_length=max_input_len)
    # if tokenizer.pad_token_id is None:
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def get_hf_dataset(task: str, model_type: str, max_input_len: int = 512):
    # load hf dataset based on task
    if task in ['sst2' ,'mrpc', 'cola', 'qnli', 'rte', 'wnli', 'stsb']:
        dataset = load_dataset('glue', task)
    elif task in ['squad', 'squad_v2']:
        dataset = load_dataset('squad')
    elif task in ['stanfordnlp/imdb']:
        dataset = load_dataset('stanfordnlp/imdb')
    else:
        raise ValueError(f'Unknown task {task}')

    # tokenize dataset
    tokenizer = get_tokenizer(model_type=model_type, 
                              max_input_len=max_input_len)
    def tokenize_function_glue(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True)

    def tokenize_function_imdb(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    if task in ['sst2' ,'mrpc', 'cola', 'qnli', 'rte', 'wnli', 'stsb']:
        tokenize_function = tokenize_function_glue
    else:
        tokenize_function = tokenize_function_imdb
    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns([s for s in dataset.column_names['train'] if s not in ['input_ids', 'attention_mask', 'label']])

    # to tensors
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return dataset


def get_sft_dataset_for_decoder(task: str, model_type: str, max_input_len: int = 512, max_output_len: int = 10):
    # load hf dataset based on task
    if task in ['sst2' ,'mrpc', 'cola', 'qnli', 'rte', 'wnli', 'stsb']:
        dataset = load_dataset('glue', task)
    elif task in ['squad', 'squad_v2']:
        dataset = load_dataset('squad')
    elif task in ['stanfordnlp/imdb']:
        dataset = load_dataset('stanfordnlp/imdb')
    else:
        raise ValueError(f'Unknown task {task}')

    # tokenize dataset
    text_tokenizer = get_tokenizer(model_type=model_type, max_input_len=max_input_len, padding_side="left")
    answer_tokenizer = get_tokenizer(model_type=model_type, max_input_len=max_output_len, padding_side="right")
    def sst2_lable2text(dataset):
        # train, dev
        label_map = {0: 'negative', 1: 'positive', -1: 'none'}

        for split in dataset:
            if split == 'train':
                df = pd.DataFrame(dataset[split])
                df['sentence'] = ['reivew: ' + s for s, l in zip(df['sentence'], df['label'])]  
                df['text_label'] = [label_map[l] for l in df['label']]
            else:
                df = pd.DataFrame(dataset[split])
                df['sentence'] = ['reivew: ' + s for s, l in zip(df['sentence'], df['label'])]  
                df['text_label'] = [label_map[l] for l in df['label']]
            dataset[split] = Dataset.from_pandas(df)
        # test first example in train and validation
        # print(dataset['train'][0])
        # print(dataset['validation'][0])
        return dataset
   
    if task == 'sst2':
        dataset = sst2_lable2text(dataset)

    def train_tokenize_function(example):
        tokenized_inputs = text_tokenizer(example['sentence'], max_length=max_input_len,padding='max_length', truncation=True, add_special_tokens=False)
        tokenized_targets = answer_tokenizer(example['text_label'], max_length=max_output_len, padding='max_length', truncation=True, add_special_tokens=False)
        
        tokenized_dataset = {}
        tokenized_dataset['input_ids'] = tokenized_inputs['input_ids']
        tokenized_dataset['attention_mask'] = tokenized_inputs['attention_mask']
        tokenized_dataset['decoder_input_ids'] = tokenized_targets['input_ids']
        tokenized_dataset['decoder_attention_mask'] = tokenized_targets['attention_mask']
        return tokenized_dataset
    
    def valid_tokenize_function(example):
        tokenized_inputs = text_tokenizer(example['sentence'], max_length=max_input_len,padding='max_length', truncation=True, add_special_tokens=False)
        tokenized_targets = answer_tokenizer(example['text_label'], max_length=max_output_len, padding='max_length', truncation=True, add_special_tokens=False)
        
        tokenized_dataset = {}
        tokenized_dataset['input_ids'] = tokenized_inputs['input_ids']
        tokenized_dataset['attention_mask'] = tokenized_inputs['attention_mask']
        tokenized_dataset['decoder_input_ids'] = tokenized_targets['input_ids']
        tokenized_dataset['decoder_attention_mask'] = tokenized_targets['attention_mask']
        return tokenized_dataset
    
    if task in ['sst2', 'mrpc', 'qnli']:
        dataset['train'] = dataset['train'].map(train_tokenize_function, batched=True)
        dataset['validation'] = dataset['validation'].map(valid_tokenize_function, batched=True)
        dataset['test'] = dataset['test'].map(valid_tokenize_function, batched=True)

        dataset = dataset.remove_columns(['sentence', 'label', 'text_label'])

    dataset.set_format("torch")
    return dataset


def get_hf_dataset_for_decoder(task: str, model_type: str, max_input_len: int = 512, max_output_len: int = 10):
    # load hf dataset based on task
    if task in ['sst2' ,'mrpc', 'cola', 'qnli', 'rte', 'wnli', 'stsb']:
        dataset = load_dataset('glue', task)
    elif task in ['squad', 'squad_v2']:
        dataset = load_dataset('squad')
    elif task in ['stanfordnlp/imdb']:
        dataset = load_dataset('stanfordnlp/imdb')
    else:
        raise ValueError(f'Unknown task {task}')

    # tokenize dataset
    text_tokenizer = get_tokenizer(model_type=model_type, max_input_len=max_input_len, padding_side="left")
    answer_tokenizer = get_tokenizer(model_type=model_type, max_input_len=max_output_len, padding_side="right")
    def sst2_lable2text(dataset):
        # train, dev
        label_map = {0: 'negative', 1: 'positive', -1: 'none'}

        for split in dataset:
            if split == 'train':
                df = pd.DataFrame(dataset[split])
                df['sentence'] = ['reivew: ' + s for s, l in zip(df['sentence'], df['label'])]  
                df['text_label'] = [label_map[l] for l in df['label']]
            else:
                df = pd.DataFrame(dataset[split])
                df['sentence'] = ['reivew: ' + s for s, l in zip(df['sentence'], df['label'])]  
                df['text_label'] = [label_map[l] for l in df['label']]
            dataset[split] = Dataset.from_pandas(df)
        # test first example in train and validation
        # print(dataset['train'][0])
        # print(dataset['validation'][0])
        return dataset
   
    if task == 'sst2':
        dataset = sst2_lable2text(dataset)

    def train_tokenize_function(example):
        tokenized_inputs = text_tokenizer(example['sentence'], max_length=max_input_len,padding='max_length', truncation=True, add_special_tokens=False)
        tokenized_targets = answer_tokenizer(example['text_label'], max_length=max_output_len, padding='max_length', truncation=True, add_special_tokens=False)
        
        tokenized_dataset = {}
        tokenized_dataset['input_ids'] = tokenized_inputs['input_ids']
        tokenized_dataset['attention_mask'] = tokenized_inputs['attention_mask']
        tokenized_dataset['decoder_input_ids'] = tokenized_targets['input_ids']
        tokenized_dataset['decoder_attention_mask'] = tokenized_targets['attention_mask']
        return tokenized_dataset
    
    def valid_tokenize_function(example):
        tokenized_inputs = text_tokenizer(example['sentence'], max_length=max_input_len,padding='max_length', truncation=True, add_special_tokens=False)
        tokenized_targets = answer_tokenizer(example['text_label'], max_length=max_output_len, padding='max_length', truncation=True, add_special_tokens=False)
        
        tokenized_dataset = {}
        tokenized_dataset['input_ids'] = tokenized_inputs['input_ids']
        tokenized_dataset['attention_mask'] = tokenized_inputs['attention_mask']
        tokenized_dataset['decoder_input_ids'] = tokenized_targets['input_ids']
        tokenized_dataset['decoder_attention_mask'] = tokenized_targets['attention_mask']
        return tokenized_dataset
    
    if task in ['sst2', 'mrpc', 'qnli']:
        dataset['train'] = dataset['train'].map(train_tokenize_function, batched=True)
        dataset['validation'] = dataset['validation'].map(valid_tokenize_function, batched=True)
        dataset['test'] = dataset['test'].map(valid_tokenize_function, batched=True)

        dataset = dataset.remove_columns(['sentence', 'label', 'text_label'])

    dataset.set_format("torch")
    return dataset


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
