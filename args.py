import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Training mode and method
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'pretrain'], required=True, help='Training mode')
    parser.add_argument('--method', type=str, required=True, help='Training method')

    # Model configuration
    parser.add_argument('--model_configs', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--model_type', type=str, default='bert-base-uncased', help='Model used for training')
    parser.add_argument('--model_ckpt', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--bf16', default=False, action='store_true', help='Use bfloat16 precision')
    parser.add_argument('--bits', type=int, default=4, help='Quantization bits')

    # Dataset configuration  
    parser.add_argument('--dataset_format', type=str, default='input-output', help='Format of the dataset')
    parser.add_argument('--dataset', type=str, default='sst2', help='Dataset name')
    parser.add_argument('--max_input_len', type=int, default=64, help='Max length of input prompt')
    parser.add_argument('--max_output_len', type=int, default=10, help='Max length of output')
    parser.add_argument('--num_side_tokens', type=int, default=10, help='Number of side tokens')

    # LoRA parameters
    parser.add_argument('--reduction_factor', type=int, default=16, help='LoRA rank reduction factor')
    parser.add_argument('--peft_hidden_size', type=int, default=64, help='PEFT adapter hidden size')
    parser.add_argument('--scaling_alpha', type=int, default=8, help='LoRA scaling factor')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--num_warmup_step_ratio', type=float, default=0.0, help='Warmup steps ratio')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')

    # System configuration
    parser.add_argument('--gpus', type=str, default='0', help='GPU indices separated by comma')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_dataloader_workers', type=int, default=8, help='Number of dataloader workers')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=2000, help='Logging interval steps')
    parser.add_argument('--save_interval', type=int, default=5000, help='Checkpoint saving interval')
    parser.add_argument('--eval_interval', type=float, default=2000, help='Evaluation interval steps')
    parser.add_argument('--nosave', default=False, action='store_true', help='Disable saving checkpoints')
    parser.add_argument('--save_dir', type=str, default='./save/', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./log', help='Directory to save logs')

    # Evaluation settings
    parser.add_argument('--do_eval', default=False, action='store_true', help='Run evaluation')
    parser.add_argument('--do_train', default=True, action='store_true', help='Run training')
    parser.add_argument('--eval_dataset_size', type=int, default=1000, help='Validation dataset size')
    parser.add_argument('--max_eval_samples', type=int, default=1000, help='Maximum evaluation samples')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    return args

