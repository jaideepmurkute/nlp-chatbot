
import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from summarizer import Summarizer
from history_manager import SummarizationHistoryManager
import torch

class TestSummarization(unittest.TestCase):
    
    def setUp(self):
        # Mock Model and Tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.device = 'cpu'
        
        # Setup Tokenizer Mocks
        self.mock_tokenizer.pad_token = None
        self.mock_tokenizer.eos_token_id = 99
        self.mock_tokenizer.apply_chat_template.return_value = "System: ... User: Summarize this..."
        
        # Mock BatchEncoding with .to()
        class MockBatchEncoding(dict):
            def to(self, device): return self
            
        mock_input = MockBatchEncoding({'input_ids': torch.tensor([[1, 2, 3]])})
        self.mock_tokenizer.side_effect = lambda text, return_tensors: mock_input
        self.mock_tokenizer.return_value = mock_input # For direct calls
        
        # Mock Decode
        self.mock_tokenizer.decode.return_value = "This is a new summary."
        
        # Mock Model Generate
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        
        # Initialize HistoryManager (which creates Summarizer internally)
        self.history_manager = SummarizationHistoryManager(
            summarizer_model=self.mock_model, 
            summarizer_tokenizer=self.mock_tokenizer, 
            device=self.device
        )
        self.summarizer = self.history_manager.summarizer # Access for direct testing if needed

    def test_summarizer_prompt_generation(self):
        """Test that the summarizer constructs the prompt correctly."""
        
        old_msgs = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello'}
        ]
        
        # We can't easily check the internal string construction without refactoring or spying, 
        # but we can verify it calls the model generate.
        
        new_summary = self.summarizer.summarize("Old Summary", old_msgs)
        
        self.assertTrue(self.mock_model.generate.called)
        self.assertEqual(new_summary, "This is a new summary.")
        
    def test_prune_history_trigger(self):
        """Test that pruning triggers when buffer limit is exceeded."""
        
        config = {'max_buffer_msgs': 2}
        
        # Create history > 2 (by reference, list will be mutated)
        full_history = [
            {'role': 'user', 'content': 'Msg 1 (Old)'}, # Should be summarized
            {'role': 'assistant', 'content': 'Msg 2 (Old)'}, # Should be summarized
            {'role': 'user', 'content': 'Msg 3 (Buffer)'},
            {'role': 'assistant', 'content': 'Msg 4 (Buffer)'}
        ]
        
        # This call should:
        # 1. Trigger summarization
        # 2. Mutate full_history (delete old msgs)
        # 3. Return prompt (System + Summary + Buffer + User Input)
        final_msgs = self.history_manager.process_messages(
            messages=full_history,
            system_prompt="System Prompt",
            user_input="Current User Input",
            config=config,
            tokenizer=self.mock_tokenizer
        )
        
        # Check call
        self.assertTrue(self.mock_model.generate.called)
        
        # Check Internal Summary State
        self.assertEqual(self.history_manager.current_summary, "This is a new summary.")
        
        # Check Mutation (Buffer Logic)
        self.assertEqual(len(full_history), 2)
        self.assertEqual(full_history[0]['content'], 'Msg 3 (Buffer)')
        
        # Check Final Output (System + Summary? + Buffer + User)
        # 0: System + Summary
        # 1: Buffer 1
        # 2: Buffer 2
        # 3: User Input
        self.assertIn("Previous Conversation Summary", final_msgs[0]['content'])
        self.assertIn("This is a new summary", final_msgs[0]['content'])
        self.assertEqual(len(final_msgs), 4)

    def test_prune_history_no_trigger(self):
        """Test that pruning DOES NOT trigger when under buffer limit."""
        
        config = {'max_buffer_msgs': 5}
        
        full_history = [
            {'role': 'user', 'content': 'Msg 1'},
            {'role': 'assistant', 'content': 'Msg 2'}
        ]
        
        # Clear mock calls to be sure
        self.mock_model.generate.reset_mock()
        
        final_msgs = self.history_manager.process_messages(
            messages=full_history,
            system_prompt="System",
            user_input="User",
            config=config,
            tokenizer=self.mock_tokenizer
        )
        
        # Should NOT call generate
        self.assertFalse(self.mock_model.generate.called)
        
        # Should remain unchanged
        self.assertEqual(len(full_history), 2)
        
        # Output: System + History (2) + User = 4
        self.assertEqual(len(final_msgs), 4)

if __name__ == '__main__':
    unittest.main()
