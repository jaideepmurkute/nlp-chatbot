
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from config import Config
from dataset import ConversationDataset
from utils import housekeeping, plot_training_logs
from inference import test


def train_epoch(cfg, model, dataloader, optimizer, scheduler, device):
    accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
    log_interval = cfg.get('log_interval', 10)
    use_gradient_clipping = cfg.get('use_gradient_clipping', True)

    model.train()
    total_loss = 0
    losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad() # Initialize gradients

    for step, batch in enumerate(progress_bar):
        # if batch == 5:
        #     break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            if use_gradient_clipping:
                # Gradient Clipping - prevent exploding grads - max l2 norm of grads allowed=1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        current_loss = loss.item() * accumulation_steps # Scale back for logging
        total_loss += current_loss
        losses.append(current_loss)
        
        if step % log_interval == 0:
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
    
    return avg_loss, perplexity


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # if batch == 5:
            #     break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
        
    return avg_loss, perplexity


def train(cfg):
    housekeeping(cfg)
    
    # --------------------------------------
    
    device = torch.device(cfg['device'])
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'])
    
    # Enable Gradient Checkpointing (saves memory)
    if cfg.get('use_gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    # Apply LoRA if enabled
    # will create a separate set of lora matrix weights and freeze older larger matrices.
    if cfg.get('use_lora', False):
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            print("Applying LoRA for memory-efficient training...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=cfg.get('lora_r', 8), # Rank of the update matrices (The "capacity" of new info provided)
                lora_alpha=cfg.get('lora_alpha', 32), # Scaling factor (Like learning rate multiplier: Scale = alpha/r)
                lora_dropout=cfg.get('lora_dropout', 0.1)
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except ImportError:
            print("Warning: 'peft' library not found. Training full model (might OOM).")
            print("Please install it: pip install peft")

    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --------------------------------------
    
    print(f"Loading data from {cfg['data_path']}...")
    df = pd.read_csv(cfg['data_path'])
    df = df[:10]

    full_dataset = ConversationDataset(tokenizer, df, cfg['max_len'])
    
    # Index-based split using sklearn
    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=cfg['seed'], shuffle=True)
    
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False)
    
    # ----------------------------------
    optimizer = AdamW(model.parameters(), lr=cfg['learning_rate'])

    total_steps = len(train_loader) * cfg['epochs'] // cfg.get('gradient_accumulation_steps', 1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['warmup_steps'],
        num_training_steps=total_steps
    )

    # ----------------------------------
    
    base_save_path = os.path.join(cfg['model_store_dir'], cfg['experiment_name'], 'finetuned_model')
    best_model_path = os.path.join(base_save_path, 'best')
    last_model_path = os.path.join(base_save_path, 'last')
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(last_model_path, exist_ok=True)
    
    best_val_loss = float('inf')
    last_best_epoch = 0

    # ----------------------------------
    print("Training...")
    training_log = {
                'train': {'loss': [], 'perplexity': []}, 
                'val': {'loss': [], 'perplexity': []}
                }
    
    for epoch in range(cfg['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        
        train_loss, train_perplexity = train_epoch(cfg, model, train_loader, optimizer, scheduler, 
                                                device)
        
        val_loss, val_perplexity = validate_epoch(model, val_loader, device)
        
        training_log['train']['loss'].append(train_loss)
        training_log['val']['loss'].append(val_loss)
        training_log['train']['perplexity'].append(train_perplexity)
        training_log['val']['perplexity'].append(val_perplexity)
        
        print(f"Loss: Train: {training_log['train']['loss'][-1]} \t \
                Val: {training_log['val']['loss'][-1]}")
        print(f"Perplexity: Train: {training_log['train']['perplexity'][-1]} \t \
                Val: {training_log['val']['perplexity'][-1]}")

        # Save latest model
        print(f"Saving latest model to {last_model_path}...")
        model.save_pretrained(last_model_path)
        tokenizer.save_pretrained(last_model_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_best_epoch = epoch
            print(f"New best model found (val_loss: {val_loss:.4f}). Saving to {best_model_path}...")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
        else:
            print(f"Not the best epoch. Last best epoch: {last_best_epoch}")
            if epoch - last_best_epoch >= cfg['early_stopping_patience']:
                print(f"No improvement in validation loss for {cfg['early_stopping_patience']} epochs.")
                print("Early stopping training.")
                break

    print("Training done!")

    # ----------------------------------
    
    plot_training_logs(cfg, training_log)    
    

if __name__ == "__main__":
    cfg = Config().get_config()

    if cfg['choice'] == 1:
        train(cfg)
    elif cfg['choice'] == 2:
        test(cfg) 
    else:
        raise ValueError(f"Invalid value for cfg[`choice`]: {cfg['choice']}")
