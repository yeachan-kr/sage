import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_DISABLED'] = 'true'

import numpy as np
import json
import copy
from tqdm.auto import tqdm

import torch
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from args import get_args
# from data import QADataset
from utils.utils import set_random_seed
from utils.data_utils import get_tokenizer

# OPT Models
from models.llama.modeling_llama import BTPLlamaForSequenceClassification
from transformers import LlamaForSequenceClassification


# from modeling_opt_qst import QSTOPTForCausalLM, OPTForCausalLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.cls_data_module import make_data_module, cls_rev_label_map
from evaluate import load
from torch.utils.data import DataLoader

from trainer.glue_trainer import GLUECLSTrainer

import torch.nn.functional as F 

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO'))
log = logging.getLogger(__name__)


def model_parameters_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    memory_bytes = total_params * 2  # assuming float32 parameters
    memory_gb = memory_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return memory_gb

def optimizer_memory(optimizer):
    memory_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                memory_bytes += v.numel() * v.element_size()
    memory_gb = memory_bytes / (1024 ** 3)  # Convert bytes to gigabytes
    return memory_gb

def main():
    args = get_args()
    for arg in vars(args):
        log.info(f'{arg}: {getattr(args, arg)}') 

    set_random_seed(args.seed)
    num_gpus = torch.cuda.device_count()
    device = torch.device('cuda' if num_gpus > 0 else 'cpu')
    log.info(f'Detected {num_gpus} GPUS')

    # Load data
    log.info('Loading data ...')
    text_tokenizer = get_tokenizer(model_type=args.model_type, max_input_len=args.max_input_len, padding_side="left")
    answer_tokenizer = get_tokenizer(model_type=args.model_type, max_input_len=args.max_output_len, padding_side="right")
    data_module = make_data_module(input_tokenizer=text_tokenizer, output_tokenizer=answer_tokenizer, args=args)

    eot = "<|eot_id|>"
    eot_id = text_tokenizer.convert_tokens_to_ids(eot)
    text_tokenizer.pad_token = eot
    text_tokenizer.pad_token_id = eot_id
    answer_tokenizer.pad_token = eot
    answer_tokenizer.pad_token_id = eot_id

    # Initialize models and optimizer
    log.info('Initializing models ...')
    if 'llama' in args.model_type.lower():
        if args.method == 'base':

            model = LlamaForSequenceClassification.from_pretrained(args.model_type,
                                                                   torch_dtype=torch.bfloat16,
                                                                   device_map="auto",
                                                                   num_labels=data_module['num_labels']).to(device)
            config = LoraConfig(
                r=args.peft_hidden_size,
                lora_alpha=args.scaling_alpha,
                target_modules='all-linear',
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, config).to(device)

        elif args.method == 'btplora':

            model = BTPLlamaForSequenceClassification.from_pretrained(args.model_type,
                                                                      num_labels=data_module['num_labels'],
                                                                      torch_dtype=torch.bfloat16)

            config = LoraConfig(
                r=args.peft_hidden_size,
                lora_alpha=args.scaling_alpha,
                target_modules='all-linear',
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, config).to(device)       
        

    if 'llama' in args.model_type.lower():
        # pad token
        if args.method != 'plora':
            model.config.pad_token = eot
            model.config.pad_token_id = eot_id
        else:
            model.model.config.pad_token = eot
            model.model.config.pad_token_id = eot_id

    # Initialize trainer
    log.info('Initializing trainer ...')
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_interval,
        num_train_epochs=args.num_epochs,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        lr_scheduler_type="linear",
        bf16=args.bf16,
        save_total_limit = 3,
    )

    trainer = GLUECLSTrainer(
        method=args.method,
        model=model,
        tokenizer=text_tokenizer,
        dataset=args.dataset,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != 'predict_dataset' and k != 'num_labels'},
    )
    
    if args.method == 'zs':
        log.info('Zero-shot evaluation ...')
        trainer.evaluate()

    if args.do_train and args.method != 'zs':
        log.info('Training ...')
        trainer.train()

if __name__ == '__main__':

    main()
