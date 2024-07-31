
from utils import get_model_and_tokenizer

class ModelSingleton:
    _instance = None
    
    def __new__(cls, cfg, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelSingleton, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialize_model(cfg)
        return cls._instance

    def initialize_model(self, cfg):
        
        # Initialize your model and tokenizer here using cfg
        self.model, self.tokenizer = get_model_and_tokenizer(cfg)
        
        