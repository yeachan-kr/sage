import os 
import copy
import argparse
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset

import transformers
from dataclasses import dataclass
from typing import Dict, Sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

cls_label_map = {
                'sst2': {0: 'negative', 1: 'positive', -1: 'none'},
                'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction', -1: 'none'},
                'cola': {0: 'unacceptable', 1: 'acceptable', -1: 'none'},
                'mrpc': {0: '0', 1: '1', -1: 'none'},
                'qnli': {0: 'entailment', 1: 'not_entailment', -1: 'none'},
                'qqp': {0: 'not duplicate', 1: 'duplicate', -1: 'none'},
                'rte': {0: '0', 1: '1', -1: 'none'},
                'wnli': {0: 'not_entailment', 1: 'not_entailment', -1: 'none'},
                'stanfordnlp/imdb': {0: 'negative', 1: 'positive', -1: 'none'},
                'SetFit/20_newsgroups': {v : str(v) for v in range(20) },
                }

cls_rev_label_map = {}
for task, label_map in cls_label_map.items():
    cls_rev_label_map[task] = {v: k for k, v in label_map.items()}

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(input_tokenizer: transformers.PreTrainedTokenizer, output_tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        elif dataset_name == 'sst2':
            dataset = load_dataset("glue", "sst2") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['sentence'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'qnli':
            dataset = load_dataset("glue", "qnli") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['question'] + ' ' + x['sentence'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'rte':
            # dataset = load_dataset("glue", "rte") # train, validation, test
            dataset = load_dataset("aps/super_glue", "rte") # train, validation, test
            dataset = dataset.map(lambda x: {'input': 'Premise: ' + x['premise'] + ' Hypothesis: ' + x['hypothesis'], 'output': x['label']})
            return dataset
        elif dataset_name == 'mrpc':
            # dataset = load_dataset("glue", "rte") # train, validation, test
            dataset = load_dataset("glue", "mrpc") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['sentence1'] + ' ' + x['sentence2'], 'output': x['label']})
            return dataset
        elif dataset_name == 'qqp':
            # dataset = load_dataset("glue", "qqp") # train, validation, test
            dataset = load_dataset("glue", "qqp") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['question1'] + ' ' + x['question2'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'mnli':
            # dataset = load_dataset("glue", "rte") # train, validation, test
            dataset = load_dataset("glue", "mnli") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['premise'] + ' ' + x['hypothesis'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'cola':
            # dataset = load_dataset("glue", "rte") # train, validation, test
            dataset = load_dataset("glue", "cola") # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['sentence'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'stanfordnlp/imdb':
            dataset = load_dataset('stanfordnlp/imdb') # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['text'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'SetFit/20_newsgroups':
            dataset = load_dataset('SetFit/20_newsgroups') # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['text'], 'output': cls_label_map[dataset_name][x['label']]})
            return dataset
        elif dataset_name == 'openai/gsm8k':
            dataset = load_dataset('openai/gsm8k', 'main') # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['question'], 'output': x['answer']})
            return dataset
        elif dataset_name == 'yelp':
            dataset = load_dataset('fancyzhx/yelp_polarity') # train, validation, test
            dataset = dataset.map(lambda x: {'input': x['text'], 'output': x['label']})
            return dataset
        
        else:
            if os.path.exists(dataset_name):
                # print(os.path.exists(dataset_name))
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
                dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
                (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

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

    data_collator = DataCollatorForCausalLM(
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        source_max_len=args.max_input_len,
        target_max_len=args.max_output_len,
        train_on_source=False,
        predict_with_generate=True,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        # predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )



@dataclass
class DataCollatorForCausalLM(object):
    input_tokenizer: transformers.PreTrainedTokenizer
    output_tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.input_tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.output_tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.input_tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
            # padding='max_length',
        )
        tokenized_targets = self.output_tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
            # padding='max_length',
        )

        # Build the input and labels for causal LM
        only_input_ids = []
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            only_input_ids.append(torch.tensor(tokenized_source))
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor(
                    [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            )        
        only_input_ids = pad_sequence(only_input_ids, batch_first=True, padding_value=self.input_tokenizer.pad_token_id)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.input_tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        # print
        data_dict = {
            'input_ids': input_ids,
            'source_len': torch.tensor(only_input_ids.size(1)),
            'attention_mask': input_ids.ne(self.input_tokenizer.pad_token_id),
        }
        if labels is not None:
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

