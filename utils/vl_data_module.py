import os 
import copy
import argparse
import pandas as pd
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset

import transformers
from dataclasses import dataclass
from typing import Dict, Sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"




def make_data_module(processor: transformers.ViltProcessor, id2label: Dict, args) -> Dict:

    # Load dataset.
    dataset = load_dataset(args.dataset)

    # Split train/eval, reduce size
    if args.do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'validation' in dataset:
            eval_dataset = dataset['validation']
        elif 'validation_matched' in dataset:
            eval_dataset = dataset['validation_matched']
        elif 'test' in dataset:
            eval_dataset = dataset['test']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        # if args.group_by_length:
        #     eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    
    if args.do_train:
        train_dataset = dataset['train']
        # if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        #     train_dataset = train_dataset.select(range(args.max_train_samples))
        # if args.group_by_length:
        #     train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForVQA(
        processor=processor,
        id2label=id2label,
        lable2id={v: k for k, v in id2label.items()}
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        # predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


@dataclass
class DataCollatorForVQA(object):
    processor: transformers.ViltProcessor
    id2label: Dict[int, str]
    lable2id: Dict[str, int]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images = [example['image'].convert('RGB') for example in instances]
        texts = [example['question'] for example in instances]

        data_dict = self.processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")
        for k, v in data_dict.items():
            data_dict[k] = v.squeeze()

        labels = torch.zeros(len(instances), len(self.id2label))
        for i, example in enumerate(instances):
            for answer in example['answers']:
                ans = answer['answer']
                if ans in self.lable2id:
                    labels[i, self.lable2id[ans]] += 1
        
        # normalizing labels
        labels = labels / (labels.sum(dim=1, keepdim=True) + 1e-6) # add epsilon to avoid division by zero
        data_dict['labels'] = labels
        return data_dict

if __name__ == '__main__':
    tokenizer1 = transformers.AutoTokenizer.from_pretrained('facebook/opt-1.3b', padding_side='left')
    tokenizer2 = transformers.AutoTokenizer.from_pretrained('facebook/opt-1.3b', padding_side='right')

    args = {
        'dataset': 'alpaca-clean',
        'max_input_len': 8,
        'max_output_len': 2,
        'train_on_source': False,
        'predict_with_generate': False,
        'do_train': True,
        'do_eval': True,
        'do_predict': False,
        'eval_dataset_size': 0.1,
        'max_train_samples': None,
        'max_eval_samples': None,
        'group_by_length': False,
        'dataset_format': 'alpaca-clean',
    } 
    args = argparse.Namespace(**args)
    data_module = make_data_module(tokenizer1, tokenizer2, args)
    print(data_module)
    # print(data_module['train_dataset'][:2])
    # print(data_module['eval_dataset'][:2])
    # batch = data_module['data_collator']([{'input': 'I love you', 'output': 'positive'}, {'input': 'I hate this', 'output': 'negative'}])
    # print(batch['input_ids'])
    # print(batch['labels'])
    # print(batch['attention_mask'])

