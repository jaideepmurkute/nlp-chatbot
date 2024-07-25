
class Config:
    def __init__(self):
        self.config = {
            'model_name': "microsoft/DialoGPT-small",
            
            'max_response_len': 512,
            'num_beams': 5,
            'num_return_sequences': 5,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.0,
        
            'seed': 42,
            'output_store_dir': "./output",
        }
        
    