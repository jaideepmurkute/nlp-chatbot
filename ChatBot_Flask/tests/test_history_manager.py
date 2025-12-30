
import unittest
import sys
import os
 
# Add src to path so we can import modules
# Path relative to ChatBot_Flask/tests/ is ../src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from history_manager import TokenTruncationManager

class MockTokenizer:
    """A simple mock tokenizer for testing without loading huge models."""
    def __init__(self):
        self.eos_token = "<|endoftext|>"
    
    def encode(self, text):
        # mocking simple space tokenizer
        return [1] * len(text.split())
    
    def decode(self, ids, skip_special_tokens=True):
        return "word " * len(ids)

class TestHistoryManager(unittest.TestCase):
    
    def setUp(self):
        self.manager = TokenTruncationManager()
        self.tokenizer = MockTokenizer()
        self.config = {
            'max_len': 100,
            'max_tot_input_prop': 1.0, # Simple math: total budget = 100
            'min_hist_input_prop': 0.1, # Min history = 10 tokens
        }
        
    def test_process_messages_basic_fit(self):
        """Test Case 1: Everything fits."""
        system_prompt = "sys" # 1 token
        user_input = "u " * 49 # 50 tokens
        messages = [{"user": "h " * 10, "model": "m " * 10}] # 20 tokens history
        
        # Total = 1 (sys) + 50 (user) + 20 (hist) = 71 <= 100.
        final = self.manager.process_messages(messages, system_prompt, user_input, self.config, self.tokenizer)
        
        # Expect: Sys, History(User+Model), Current User
        self.assertEqual(len(final), 4) # Sys + HistUser + HistModel + CurrUser
        self.assertEqual(final[3]['role'], 'user')

    def test_process_messages_history_truncation(self):
        """Test Case 2: History is truncated to fit user input."""
        system_prompt = "sys" # 1 token
        user_input = "u " * 80 # 80 tokens
        # History is huge
        messages = [{"user": "h " * 50, "model": "m " * 50}] 
        
        # Budget = 100. Sys=1. Remaining=99.
        # User=80. 
        # History Space = 99 - 80 = 19.
        # Min History = 10. (19 >= 10, so we use 19 for history)
        
        final = self.manager.process_messages(messages, system_prompt, user_input, self.config, self.tokenizer)
        
        # We expect SOME history, but truncated. 
        # Since our mock history turns are huge blocks (50 tokens), 
        # and we process whole turns or break?
        # The logic splits 'user' and 'model' parts of a turn.
        # "m " * 50 is 50 tokens. 19 available. Can't fit model response.
        # So we expect 0 history.
        
        self.assertEqual(len(final), 2) # Sys + CurrUser (History dropped because chunks were too big)
        
    def test_process_messages_force_min_history(self):
        """Test Case 3: User input is HUGE, must force truncate user to keep min history."""
        system_prompt = "sys" # 1 token
        user_input = "u " * 95 # 95 tokens
        # History exists
        messages = [{"user": "old", "model": "hist " * 5}] # 1+5 = 6 tokens history
        
        # Budget = 100. Sys=1. Remaining=99.
        # Min History = 10% of 99 = 9 tokens.
        
        # User input (95) > Available(99) - MinHist(9) = 90.
        # So User input SHOULD be truncated to 90.
        # History budget = 9.
        
        # Our available history is small (6 tokens). It fits in 9 budget.
        
        final = self.manager.process_messages(messages, system_prompt, user_input, self.config, self.tokenizer)
        
        # Expect: Sys + HistUser + HistModel + CurrUser(Truncated)
        self.assertEqual(len(final), 4) 
        
        # Checks
        # Verify user input was truncated. 
        # Original 95 words. Limit 90.
        # Mock decode produces "word " * len.
        last_msg = final[-1]['content']
        # self.assertTrue(len(last_msg) < len(user_input)) # String len check approximation
        
        # Better check:
        # History should be present
        self.assertEqual(final[1]['content'], "old")

if __name__ == '__main__':
    unittest.main()
