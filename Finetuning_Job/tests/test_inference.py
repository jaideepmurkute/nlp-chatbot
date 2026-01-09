
import unittest
import sys
import os
import shutil
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from inference import test

class TestInferencePipeline(unittest.TestCase):
    
    def setUp(self):
        # Setup Validation Directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_outputs_inference')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Mock Config
        self.cfg = {
            'choice': 2, # Inference
            'experiment_name': 'test_experiment',
            'model_name': 'Qwen/Qwen2.5-0.5B-Instruct', 
            'data_path': 'dummy_test.csv',
            'model_store_dir': self.test_dir,
            
            # Key Params for load_model
            'checkpoint_type': 'best',
            'device': 'cpu',
            'use_4bit': False, 
            
            # Generation Params
            'max_len': 32
        }
        
    def tearDown(self):
        # Cleanup
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('pandas.read_csv')
    @patch('inference.AutoModelForCausalLM')
    @patch('inference.AutoTokenizer')
    def test_inference_loop_smoke(self, mock_tokenizer, mock_model_cls, mock_read_csv):
        """
        Smoke Test: Runs the inference.test() function with heavily mocked components.
        Verifies that it loads data, generates responses, and saves a CSV.
        """
        
        # 1. Mock Data Loading
        mock_read_csv.return_value = pd.DataFrame({
            'instruction': ['Tell me a joke', 'What is AI?'],
            'response': ['Haha', 'Machines learning']
        })
        
        # 2. Mock Tokenizer
        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = '</s>'
        mock_tok_instance.eos_token_id = 99
        # Mock apply_chat_template to return a simple string
        mock_tok_instance.apply_chat_template.return_value = "System: ... User: ... Assistant:"
        
        # Helper class to mock BatchEncoding which has .to() method
        class MockBatchEncoding(dict):
            def to(self, device):
                return self

        mock_encoded = MockBatchEncoding({
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        })

        # Mock __call__ (encoding) to return our custom object
        # inference.py calls tokenizer(text, return_tensors="pt")
        # We handle both side_effect and return_value to be safe
        mock_tok_instance.side_effect = lambda text, return_tensors: mock_encoded
        mock_tok_instance.return_value = mock_encoded
        
        # Mock decode
        mock_tok_instance.decode.return_value = "Mocked Response"
        
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance
        
        # 3. Mock Model
        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        
        # Mock generate()
        # inference.py: outputs = model.generate(...)
        # It expects a tensor of shape [batch, seq_len]
        # input len is 3. We return something longer to simulate generation.
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        # 4. Run Test
        try:
            test(self.cfg)
        except Exception as e:
            self.fail(f"Inference loop crashed with error: {e}")
            
        # 5. Verify Output
        # Expected output file: model_store/experiment_name/test_predictions_best.csv
        # Our cfg['model_store_dir'] is self.test_dir.
        # So path is: self.test_dir / test_experiment / test_predictions_best.csv
        
        exp_dir = os.path.join(self.test_dir, 'test_experiment')
        output_file = os.path.join(exp_dir, 'test_predictions_best.csv')
        
        self.assertTrue(os.path.exists(output_file), f"Output CSV not found at {output_file}")
        
        # Verify content
        # CANNOT use pd.read_csv because it is MOCKED to return input data!
        # Use simple string check
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print(f"Output File Content:\n{content}") # For debug
        
        # Check for header
        self.assertIn('generated_response', content)
        # Check for values
        self.assertIn('Mocked Response', content)

if __name__ == '__main__':
    unittest.main()
