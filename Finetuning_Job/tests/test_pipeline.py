
import unittest
import sys
import os
import shutil
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from main import train

class TestFinetuningPipeline(unittest.TestCase):
    
    def setUp(self):
        # Setup Validation Directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_outputs')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Mock Config
        self.cfg = {
            'choice': 1,
            'experiment_name': 'test_experiment',
            'model_name': 'Qwen/Qwen2.5-0.5B-Instruct', # Use small model or mock
            'data_path': 'dummy.csv',
            'output_dir': self.test_dir,
            'model_store_dir': self.test_dir,
            
            # Training Params (Minimal for speed)
            'epochs': 1,
            'batch_size': 2,
            'gradient_accumulation_steps': 1,
            'learning_rate': 1e-4,
            'warmup_steps': 0,
            'max_len': 32, # Very short blocks
            'seed': 42,
            'device': 'cpu', # Force CPU for testing environment compatibility
            'early_stopping_patience': 1,
            
            # Features
            'use_lora': True, # Enable LoRA to test PEFT logic integration
            'use_4bit': False, # No GPU
            'use_gradient_checkpointing': False,
            'use_gradient_clipping': False,
            
            # W&B Logic
            'use_wandb': False, # Important: Disable W&B
            'log_interval': 1
        }
        
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('pandas.read_csv')
    @patch('main.ConversationDataset')
    @patch('main.AutoModelForCausalLM')
    @patch('main.AutoTokenizer')
    def test_train_loop_smoke(self, mock_tokenizer, mock_model_cls, mock_dataset_cls, mock_read_csv):
        """
        Smoke Test: Runs the train() function with heavily mocked components 
        to verify the LOOP logic holds together without crashing.
        """
        
        mock_peft = MagicMock()
        mock_peft.LoraConfig.return_value = MagicMock()
        # CRITICAL: Return the SAME model so our mock setup (parameters, etc.) is preserved
        mock_peft.get_peft_model.side_effect = lambda model, config: model
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"
        
        with patch.dict(sys.modules, {'peft': mock_peft}):
             self._run_train_logic(mock_tokenizer, mock_model_cls, mock_dataset_cls, mock_read_csv)

    def _run_train_logic(self, mock_tokenizer, mock_model_cls, mock_dataset_cls, mock_read_csv):
        """Core logic separated to allow context manager for mocks"""
        
        # 1. Mock Data Loading
        # Create dummy dataframe
        mock_read_csv.return_value = pd.DataFrame({
            'instruction': ['hi', 'hello'],
            'response': ['bye', 'goodbye']
        })
        
        # 2. Mock Tokenizer
        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = '</s>'
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        
        # 3. Mock Dataset & Dataloader
        # The main.py creates Subset objects. We need to ensure len(dataset) works.
        mock_ds_instance = MagicMock()
        mock_ds_instance.__len__.return_value = 10 
        
        # DataLoader calls __getitem__. It MUST return Tensors for default_collate
        def get_item_side_effect(idx):
            return {
                'input_ids': torch.tensor([1, 2, 3]),
                'attention_mask': torch.tensor([1, 1, 1]),
                'labels': torch.tensor([1, 2, 3])
            }
        mock_ds_instance.__getitem__.side_effect = get_item_side_effect
        
        mock_dataset_cls.return_value = mock_ds_instance
        
        # 4. Mock Model
        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        
        # MOCK PARAMETERS for Optimizer
        # Optimizer needs iterable of tensors, and checks type strictly
        mock_param = torch.nn.Parameter(torch.tensor([0.0]))
        mock_model_instance.parameters.return_value = [mock_param]
        
        # Mock forward pass output
        mock_output = MagicMock()
        # Use a real Tensor for loss so division and .item() work correctly
        # requires_grad=True is needed for .backward() to not error (though it won't do anything meaningful without a graph)
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        mock_model_instance.return_value = mock_output
        
        # 5. Run Train
        # logic in main.py:
        #   df = pd.read_csv...
        #   full_dataset = ConversationDataset...
        #   train_loader = ...
        #   model = AutoModel...
        #   optimizer...
        #   loop...
        
        try:
            train(self.cfg)
        except Exception as e:
            self.fail(f"Training loop crashed with error: {e}")
            
        # 6. Verify Artifact Creation
        # Check if directories were created (main.py creates directories using utils)
        exp_dir = os.path.join(self.test_dir, 'test_experiment')
        self.assertTrue(os.path.exists(exp_dir))
        
        # Verify model was saved (mocked save_pretrained called?)
        # Since we mocked the model, no actual file is written by save_pretrained.
        # But we can verify correctly that the METHOD was called.
        self.assertTrue(mock_model_instance.save_pretrained.called)
        self.assertTrue(mock_tok_instance.save_pretrained.called)

if __name__ == '__main__':
    unittest.main()
