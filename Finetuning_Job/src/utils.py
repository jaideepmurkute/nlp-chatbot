
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import json

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_paths_dirs(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    experiment_dir = os.path.join(cfg['model_store_dir'], cfg['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)


def housekeeping(cfg):
    set_seeds(cfg['seed'])
    
    if cfg['choice'] == 1:
        create_paths_dirs(cfg)


def plot_training_logs(cfg, training_log):
    logs_dir = os.path.join(cfg['model_store_dir'], cfg['experiment_name'], 'finetuning_logs')
    os.makedirs(logs_dir, exist_ok=True)

    # --------------------------------
    
    training_log_save_path = os.path.join(logs_dir, 'training_log.json')
    with open(training_log_save_path, 'w') as f:
        json.dump(training_log, f)
    print(f"Training log saved to: {training_log_save_path}")
    
    # --------------------------------
    train_loss_save_path = os.path.join(logs_dir, 'train_loss.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['train']['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(train_loss_save_path)
    plt.close()
    print(f"Training loss log plots saved to: {train_loss_save_path}")
    
    # --------------------------------
    
    val_loss_save_path = os.path.join(logs_dir, 'validation_loss.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['val']['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.savefig(val_loss_save_path)
    plt.close()
    print(f"Validation loss log plots saved to: {val_loss_save_path}")

    # --------------------------------
    
    aggr_loss_save_path = os.path.join(logs_dir, 'aggregated_loss.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['train']['loss'], label='Train', color='blue')
    plt.plot(training_log['val']['loss'], label='Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(aggr_loss_save_path)
    plt.close()
    print(f"Loss log plots saved to: {aggr_loss_save_path}")

    # --------------------------------
    
    train_perplexity_save_path = os.path.join(logs_dir, 'train_perplexity.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['train']['perplexity'])
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity')
    plt.savefig(train_perplexity_save_path)
    plt.close()
    print(f"Training perplexity log plots saved to: {train_perplexity_save_path}")

    # --------------------------------
    
    val_perplexity_save_path = os.path.join(logs_dir, 'validation_perplexity.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['val']['perplexity'])
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.savefig(val_perplexity_save_path)
    plt.close()
    print(f"Validation perplexity log plots saved to: {val_perplexity_save_path}")

    # --------------------------------
    
    aggr_perplexity_save_path = os.path.join(logs_dir, 'aggregated_perplexity.png')
    plt.figure(figsize=(10, 5))
    plt.plot(training_log['train']['perplexity'], label='Train', color='blue')
    plt.plot(training_log['val']['perplexity'], label='Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(aggr_perplexity_save_path)
    plt.close()
    print(f"Perplexity log plots saved to: {aggr_perplexity_save_path}")

    # --------------------------------
    