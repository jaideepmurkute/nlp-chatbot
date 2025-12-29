
'''
This file contains the configuration parameters for the chatbot model.
'''

import os

class Config:
    def __init__(self):
        self.config = {
            # "Qwen/Qwen2.5-1.5B-Instruct", #"experiment_1/finetuned_model_dialoGPT/best", # "microsoft/DialoGPT-small"
            'model_name': "experiment_1_ft_qwen2_5-1_5B-Instruct/best",
            
            'max_convs': 50, # max no. of conversations in a single session
            
            # input + output length
            'max_len': 1000,  # must be <= model.config.n_ctx
            
            # define upper ceiling for history/input/output tokens; as proportion of max_op_len
            # leftover will be for output tokens 
            'max_tot_input_prop': 0.8,  
            
            # define upper ceiling for history tokens length; as proportion of total input length
            # leftover will be current prompt tokens
            'max_hist_input_prop': 0.8, 
            # define lower ceiling for history tokens length; as proportion of total input length
            'min_hist_input_prop': 0.1, 
            
            # output sampling parameters
            'num_beams': 1,
            'num_return_sequences': 1,
            'temperature': 0.75,         # Balanced creativity
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.0,  # Disabled to fix gibberish
            'no_repeat_ngram_size': 3,  # Prevents repeating 3-word phrases (fixes loops)
        
            'seed': 42,
            
            'output_store_dir': os.path.join('..', 'outputs'), 
            'model_store_dir': os.path.join('..', 'model_store'), 
        }
    
    def get_config(self):
        return self.config
    
        
    