
import os
import torch

class Config:
    def __init__(self):
        self.config = {
            'choice': 1, # 1: train; 2: test
            'experiment_name': 'experiment_1', 

            'model_name': "Qwen/Qwen2.5-0.5B-Instruct", # "Qwen/Qwen2.5-1.5B-Instruct", # "microsoft/DialoGPT-small",

            # 'data_path': os.path.join('..', 'data', 'sample_dataset.csv'),            
            'data_path': os.path.join('..', 'data', 'Bitext-customer-support-llm-chatbot-training-dataset.csv'),

            'output_dir': os.path.join('..', 'outputs'),
            'model_store_dir': os.path.join('..', 'model_store'),
            'checkpoint_type': 'best', # 'best' or 'last'; For testing mode only (choice=2)
            
            # Training parameters
            'epochs': 1,
            'early_stopping_patience': 1,
            'batch_size': 8, # Reduced for CPU RAM
            'gradient_accumulation_steps': 4, # Simulate larger batch size
            'use_gradient_checkpointing': True,
            'use_gradient_clipping': True,
            

            # LoRA Parameters
            'use_lora': True,
            'lora_r': 8,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            
            # Quantization Configuration (QLoRA)
            # Note: bitsandbytes requires a GPU (CUDA) and is best supported on Linux in general.
            # Windows support is experimental and manual installation might be needed. 
            'use_4bit': False,                           # Set to True to enable 4-bit quantization (Requires bitsandbytes)
            'bnb_4bit_compute_dtype': "float16",         # Computation type: DataType used for separate linear layer computations.
                                                         # float16 / bfloat16 (brain float 16 - works only on newer GPUs, 
                                                         # superior/stable in training) / float32
            'bnb_4bit_quant_type': "nf4",                # Quantization type: "nf4" (Normal Float 4) or "fp4" (Regular IEEE format, 
                                                         # but nf4 tends to work better))
            'bnb_4bit_use_double_quant': True,           # Nested quantization for memory savings
            
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
