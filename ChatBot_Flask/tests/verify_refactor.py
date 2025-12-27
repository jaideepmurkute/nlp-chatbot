
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# We need to mock CFG and ModelSingleton before importing ChatBot because it imports them at top level
sys.modules['CFG'] = MagicMock()
sys.modules['model_singleton'] = MagicMock()
sys.modules['utils'] = MagicMock()

from chatbot import ChatBot
from history_manager import ConversationContext, TokenTruncationManager

class TestChatBotRefactor(unittest.TestCase):
    def setUp(self):
        self.mock_cfg = {'session_id': 'test_session', 'model_name': 'test_model', 'max_len': 100, 'max_tot_input_prop': 0.8}
        self.mock_app = MagicMock()
        
        # Setup ModelSingleton mock
        self.mock_model_singleton = MagicMock()
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_model_singleton.model = self.mock_model
        self.mock_model_singleton.tokenizer = self.mock_tokenizer
        
        # Mock Tokenizer behavior
        self.mock_tokenizer.encode_plus.return_value = {
            'input_ids': torch.tensor([[101, 102]]), 
            'attention_mask': torch.tensor([[1, 1]])
        }
        self.mock_tokenizer.eos_token = "</s>"
        self.mock_tokenizer.eos_token_id = 2
        self.mock_tokenizer.decode.return_value = "Test Response"
        
        # Patching ModelSingleton class in chatbot module
        with patch('chatbot.ModelSingleton', return_value=self.mock_model_singleton), \
             patch('chatbot.Config', return_value=MagicMock(config=self.mock_cfg)), \
             patch('chatbot.create_dirs_paths', return_value=self.mock_cfg), \
             patch('chatbot.save_config'), \
             patch('chatbot.save_conversations'):
             
            self.bot = ChatBot(self.mock_cfg, self.mock_app)

    def test_structure(self):
        print("\nTesting Class Structure...")
        self.assertIsInstance(self.bot.context, ConversationContext)
        self.assertIsInstance(self.bot.history_manager, TokenTruncationManager)
        print("Structure Confirmed: ChatBot has ConversationContext and TokenTruncationManager.")

    def test_delegation(self):
        print("\nTesting History Delegation...")
        # Mock the process method to ensure it's called
        with patch.object(self.bot.history_manager, 'process', wraps=self.bot.history_manager.process) as mock_process:
            self.bot.generate_response("Hello")
            
            mock_process.assert_called_once()
            print("Delegation Confirmed: generate_response called history_manager.process().")
            
            # Check if logs were updated
            self.assertEqual(len(self.bot.context.logs), 2) # Initial log + 1 interaction
            self.assertEqual(self.bot.context.logs[-1]['user'], "Hello")
            print("State Update Confirmed: Logs updated correctly.")

if __name__ == '__main__':
    unittest.main()
