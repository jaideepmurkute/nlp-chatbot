
import os
import torch

class Config:
    def __init__(self):
        self.config = {
            'choice': 1, # 1: train; 2: test
            'experiment_name': 'experiment_1', 

            'model_name': "Qwen/Qwen2.5-1.5B-Instruct", # "microsoft/DialoGPT-small",
            'data_path': os.path.join('..', 'data', 'sample_dataset.csv'),
            'output_dir': os.path.join('..', 'outputs'),
            'model_store_dir': os.path.join('..', 'model_store'),
            'checkpoint_type': 'best', # 'best' or 'last'; For testing mode only (choice=2)
            
            # Training parameters
            'epochs': 1,
            'early_stopping_patience': 3,
            'batch_size': 1, # Reduced for CPU RAM
            'gradient_accumulation_steps': 4, # Simulate larger batch size
            
            # LoRA Parameters
            'use_lora': True,
            'lora_r': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            
            'learning_rate': 2e-4, # LoRA usually needs higher LR
            'warmup_steps': 100,
            'max_len': 512, # Max sequence length
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Logging
            'log_interval': 10,
        }

    def get_config(self):
        return self.config
