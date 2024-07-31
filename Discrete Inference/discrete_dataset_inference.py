import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


class DiscreteInference:
    def __init__(self, config):
        self.config = config
        
    def __create_paths_dirs(config):
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])
    
    def __get_model_and_tokenizer(self, config):
        model = AutoModelForCausalLM.from_pretrained(self.config["model_name"])
        
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
        
        return model, tokenizer   

    def __save_responses(self, dataset_df, responses):
        assert os.path.exists(self.config['output_dir']), f"Output directory not found - {self.config['output_dir']}"
            
        op_df = pd.DataFrame({'input': dataset_df['input'], 'response': responses})
        
        output_fpath = os.path.join(self.config['output_dir'], 'responses.csv')
        op_df.to_csv(output_fpath, index=False)
    
    def __load_dataset(self):
        assert os.path.exists(self.config['data_dir']), f"Data directory not found - {self.config['data_dir']}"
        
        data_fpath = os.path.join(self.config["data_dir"], self.config["dataset_fname"])
        assert os.path.exists(data_fpath), "Dataset file not found..."
        
        ftype = data_fpath.split('.')[-1]
        assert ftype in ['csv'], "Dataset file format not supported..."
        
        print("Loading data file from path: ", data_fpath)
        
        if ftype == 'csv':
            dataset = pd.read_csv(data_fpath)
        else:
            print("Dataset file format not supported...")
            sys.exit(1)
            
        return dataset
    
    def __encode_dataset(self, dataset, tokenizer):
        # By default; the tokenizer will pad to the length of the longest sequence in the batch, 
        # not necessarily to model_max_length.
        def tokenize_fn(text): return tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        tokenized_data = dataset['input'].apply(tokenize_fn)
        
        # Convert encoded_data to the correct format
        encoded_inputs = []
        for item in tokenized_data:
            encoded_inputs.append({'input_ids': item['input_ids'].squeeze(0), \
                                'attention_mask': item['attention_mask'].squeeze(0)})
        
        return encoded_inputs
    
    def __generate_response(self, encoded_data, model, tokenizer):
        # Create a Trainer instance
        training_args = TrainingArguments(output_dir=self.config["output_dir"],
                                        per_device_eval_batch_size=self.config['batch_size'])
        
        # collate_fn to pad the input tensors to the same length
        data_collator = DataCollatorWithPadding(tokenizer)
        trainer = Trainer(model=model, args=training_args, data_collator=data_collator)
        
        # op_logits.predictions.shape: (batch_size, seq_len, vocab_size)
        '''
        The model produces an output (logits) for each input token position, including padding tokens.
        This is why the seq_len in the output matches the input sequence length after padding.
        '''
        op_logits = trainer.predict(encoded_data).predictions
        op_probs = F.softmax(torch.tensor(op_logits), dim=-1)
        
        # op_cls.shape = (batch_size, seq_len)
        op_cls = torch.argmax(op_probs, dim=-1)
        
        responses = tokenizer.batch_decode(op_cls.tolist(), skip_special_tokens=True)
        
        return responses

    def infer(self):
        self.__create_paths_dirs(self.config)
        
        model, tokenizer = self.__get_model_and_tokenizer(self.config)

        dataset = self.__load_dataset()
        encoded_data = self.__encode_dataset(dataset, tokenizer)

        responses = self.generate_response(encoded_data, model, tokenizer)
        
        self.__save_responses(dataset, responses)
    
    


