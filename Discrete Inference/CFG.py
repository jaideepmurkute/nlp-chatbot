
class Config:
    def __init__(self) -> None:
        self.config = {
                "model_name": "microsoft/DialoGPT-medium",
                'dataset_fname': 'conv_ai_1.csv',
                'batch_size': 8, 

                "data_dir": './data',
                'output_dir': './results', 
            }
    