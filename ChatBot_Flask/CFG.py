
class Config:
    def __init__(self):
        self.config = {
            'model_name': "microsoft/DialoGPT-small",
            'max_convs': 30, # number of conversations
            
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
            
            'num_beams': 5,
            'num_return_sequences': 5,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.0,
        
            'seed': 42,
            'output_store_dir': ".\outputs",
            'model_store_dir': ".\model_store",
        
        }
    
    def get_config(self):
        return self.config
    
        
    