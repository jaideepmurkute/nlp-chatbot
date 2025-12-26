
import pandas as pd
import torch
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    """
    Custom Dataset class for loading conversation data.
    Expected format: CSV with 'context' and 'response' columns.
    """
    def __init__(self, tokenizer, data, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = str(row['context'])
        response = str(row['response'])
        
        # DialoGPT format: context + eos + response + eos
        # Note: We treat the whole sequence as input (context + response) during training.
        # The model is auto-regressive.
        
        line = context + self.tokenizer.eos_token + response + self.tokenizer.eos_token
        
        encoding = self.tokenizer(
            line,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are same as input_ids for Causal LM loss
        labels = input_ids.clone()
        
        # We want to ignore padding in the loss calculation
        # tokenizer.pad_token_id is usually same as eos_token_id for DialoGPT, but let's be safe
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
