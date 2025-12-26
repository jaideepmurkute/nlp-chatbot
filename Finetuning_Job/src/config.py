
import os
import torch

class Config:
    def __init__(self):
        self.config = {
            'choice': 2, # 1: train; 2: test
            'experiment_name': 'experiment_1', 

            'model_name': "microsoft/DialoGPT-small",
            'data_path': os.path.join('..', 'data', 'sample_dataset.csv'),
            'output_dir': os.path.join('..', 'outputs'),
            'model_store_dir': os.path.join('..', 'model_store'),
            'checkpoint_type': 'best', # 'best' or 'last'; For testing mode only (choice=2)
            
            # Training parameters
            'epochs': 1,
            'early_stopping_patience': 3,
            'batch_size': 4,
            'learning_rate': 5e-5,
            'warmup_steps': 100,
            'max_len': 512, # Max sequence length
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # Logging
            'log_interval': 10,
        }

    def get_config(self):
        return self.config
