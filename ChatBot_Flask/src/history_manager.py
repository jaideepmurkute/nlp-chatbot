"""
This module defines the strategy pattern components for ChatBot history management.

Components:
    - ConversationContext: A dataclass holding the state of the conversation (ids, masks, logs).
    - HistoryManager: An abstract base class defining the strategy interface.
    - TokenTruncationManager: A concrete implementation of HistoryManager using the original truncation logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class ConversationContext:
    """
    Holds the state of the current conversation.
    
    Attributes:
        history_ids (torch.Tensor): Tensor containing the input IDs of the conversation history.
        attention_mask (torch.Tensor): Tensor containing the attention mask for the history.
        system_ids (torch.Tensor): Tensor containing the system prompt input IDs.
        system_att_mask (torch.Tensor): Tensor containing the system prompt attention mask.
        logs (List[Dict]): A list of dictionaries logging the user-model interaction history.
    """
    history_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    attention_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    system_ids: torch.Tensor = field(default_factory=lambda: torch.tensor([[]]))
    system_att_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([[]]))
    logs: List[Dict] = field(default_factory=list)
    session_metadata: Dict = field(default_factory=dict)

class HistoryManager(ABC):
    """
    Abstract Base Class for History Management Strategies.
    
    Subclasses must implement the `process` method to define how history is merged, 
    truncated, or summarization is applied.
    """
    
    @abstractmethod
    def process(self, context: ConversationContext, user_input_enc: dict, config: dict) -> None:
        """
        Modifies the context in-place based on the user input and the specific strategy logic.

        Args:
            context (ConversationContext): The current conversation state.
            user_input_enc (dict): The tokenized user input (must contain 'input_ids' and 'attention_mask').
            config (dict): The configuration dictionary containing limits like 'max_len', 'max_tot_input_prop', etc.
        """
        pass

    @abstractmethod
    def process_messages(self, messages: List[Dict], system_prompt: str, user_input: str, config: dict, tokenizer) -> List[Dict]:
        """
        Processes a list of raw messages (dicts) and returns a pruned list that fits within context limits.
        
        Args:
            messages (List[Dict]): The full conversation history (excluding system prompt).
            system_prompt (str): The raw system prompt.
            user_input (str): The current user input.
            config (dict): Configuration dictionary.
            tokenizer: The tokenizer for counting tokens.
            
        Returns:
            List[Dict]: The final list of messages (System + History + User) ready for templating.
        """
        pass

class TokenTruncationManager(HistoryManager):
    """
    Concrete Strategy implementing the original Token Truncation logic.
    
    This strategy manages history by:
    1. Calculating the total length of history + current input.
    2. Reserving space for the system prompt and generation output.
    3. Truncating the oldest history tokens if the limit is exceeded.
    4. Truncating the current input if even the minimum history requirement cannot be met.
    
    This class supports TWO Data Pipelines:
    A. `process`: [LEGACY] Handles raw `torch.Tensor` concatenation (e.g., for DialoGPT).
    B. `process_messages`: [MODERN] Handles structured `List[Dict]` pruning (e.g., for Qwen/Llama Chat Templates).
    """
    
    def process(self, context: ConversationContext, user_input_enc: dict, config: dict) -> None:
        """
        [LEGACY PIPELINE] Merges user input into context tensors with intelligent truncation.
        
        Logic Overview:
        - Works directly on `input_ids` tensors.
        - Calculates total length (History + User Input).
        - If Total > Limit:
            1. Calculates a 'History Budget' (max_tot_input_prop) vs 'Input Budget'.
            2. Truncates history from the left (oldest tokens) to fit the budget.
            3. If user input is huge, ensures at least `min_hist_input_prop` of history is kept 
               by truncating the user input itself.
        - Updates `context.history_ids` and `context.attention_mask` in-place.
        """
        
        # Ensure context tensors are on the correct device (if input is on a device)
        # For simplicity, we assume everything is CPU or handled by PyTorch's default behavior unless advanced
        
        curr_ip_len = user_input_enc['input_ids'].shape[-1]
        
        # Handle initial case
        if context.history_ids.numel() == 0:
            context.history_ids = user_input_enc['input_ids']
            context.attention_mask = user_input_enc['attention_mask']
            return

        curr_hist_len = context.history_ids.shape[-1]
        curr_tot_ip_len = curr_ip_len + curr_hist_len
        
        # DEDUCT SYSTEM PROMPT LENGTH TO RESERVE SPACE
        system_len = context.system_ids.shape[-1] if context.system_ids.numel() > 0 else 0
        max_permissible_ip_tokens = int(config['max_len'] * config['max_tot_input_prop']) - system_len
        
        # If total length falls within limits, no truncation needed
        if curr_tot_ip_len <= max_permissible_ip_tokens:
            # Append directly
            context.history_ids = torch.cat([context.history_ids, user_input_enc['input_ids']], dim=-1) 
            context.attention_mask = torch.cat([context.attention_mask, user_input_enc['attention_mask']], dim=-1)
        else:
            # Calculate minimum history to sustain (e.g. 10% of current history)
            min_hist_needed = int(curr_hist_len * config['min_hist_input_prop'])
            
            # Calculate how much space is left for history if we keep the full input
            space_for_history = max_permissible_ip_tokens - curr_ip_len
            
            new_hist_len = 0
            new_ip_len = curr_ip_len
            
            if space_for_history >= min_hist_needed:
                # Scenario: We can fit the full input and at least the minimum history.
                new_hist_len = space_for_history
            else:
                # Scenario: Input is so large that we can't even fit minimum history.
                # Priority: Maintain minimum history, truncate input.
                new_hist_len = min_hist_needed
                new_ip_len = max_permissible_ip_tokens - new_hist_len
            
            # Apply Truncation to History
            if new_hist_len > 0:
                context.history_ids = context.history_ids[:, -new_hist_len:]
                context.attention_mask = context.attention_mask[:, -new_hist_len:]
            else:
                # Safe fallback
                context.history_ids = torch.tensor([[]])
                context.attention_mask = torch.tensor([[]])

             # Apply Truncation to Input if needed
            if new_ip_len < curr_ip_len:
                user_input_enc['input_ids'] = user_input_enc['input_ids'][:, :new_ip_len]
                user_input_enc['attention_mask'] = user_input_enc['attention_mask'][:, :new_ip_len]
            
            # Merge
            context.history_ids = torch.cat([context.history_ids, user_input_enc['input_ids']], dim=-1) 
            context.attention_mask = torch.cat([context.attention_mask, user_input_enc['attention_mask']], dim=-1)

    def process_messages(self, messages: List[Dict], system_prompt: str, user_input: str, config: dict, tokenizer) -> List[Dict]:
        """
        [MODERN PIPELINE] Implement truncation for structured message lists (Instruct Models).
        
        Logic Overview:
        - Instead of slicing tensors, this selects WHOLE messages from history.
        - Preserves the Chat Template structure (System -> History -> User).
        - Logic:
            1. Reserve space for System Prompt (Mandatory).
            2. Reserve space for Current User Input.
            3. Fill remaining space (Budget) with History messages, starting from the NEWEST.
            4. If a history message doesn't fit, stop adding (Strict Pruning).
            5. If User Input itself is too big for the context, truncate it.
        """
        final_messages = []
        
        # 1. System Prompt Token Count
        # (We treat system prompt as mandatory)
        sys_tokens = len(tokenizer.encode(system_prompt))
        
        # 2. Add System Prompt to final list first
        final_messages.append({"role": "system", "content": system_prompt})
        
        # 3. User Input Token Count
        user_tokens = len(tokenizer.encode(user_input))
        
        # 4. Calculate Budget
        max_total_tokens = int(config.get('max_len', 1000) * config.get('max_tot_input_prop', 0.8))
        
        # Space remaining for history + user input after system prompt
        available_context = max_total_tokens - sys_tokens
        
        if available_context <= 0:
            # Fallback: System prompt is too huge, just return system + user (truncated)
            # This is an edge case.
             final_messages.append({"role": "user", "content": user_input}) # Template might handle truncation or error
             return final_messages

        # If current input fits easily
        if user_tokens <= available_context:
            history_budget = available_context - user_tokens
            
            # Select history from NEWEST to OLDEST
            # Iterate backwards
            selected_history = []
            current_hist_tokens = 0
            
            for turn in reversed(messages):
                # A single turn in logs is {'user': '...', 'model': '...'}
                # We are traversing BACKWARDS: so we see Model response first, then User input
                
                # 1. Process Assistant Response
                if 'model' in turn:
                    content = turn['model']
                    turn_len = len(tokenizer.encode(content))
                    if current_hist_tokens + turn_len <= history_budget:
                        selected_history.insert(0, {"role": "assistant", "content": content})
                        current_hist_tokens += turn_len
                    else:
                        break # Stop if we can't fit the most recent half of the turn

                # 2. Process User Input
                if 'user' in turn:
                    content = turn['user']
                    turn_len = len(tokenizer.encode(content))
                    if current_hist_tokens + turn_len <= history_budget:
                        selected_history.insert(0, {"role": "user", "content": content})
                        current_hist_tokens += turn_len
                    else:
                        break # Stop
            
            final_messages.extend(selected_history)
            final_messages.append({"role": "user", "content": user_input})
            
        else:
            # Scenario: User input itself is larger than available context (minus system prompt)
            # Strategy: Truncate user input to fit. No history.
            
            # We need to decode-encode to slice text properly at token boundaries if we want to be precise,
            # or just slice tokens.
            # But here arguments are strings.
            
            # Simple approach: Encode, Slice, Decode
            enc = tokenizer.encode(user_input)
            truncated_ids = enc[:available_context]
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
            
            final_messages.append({"role": "user", "content": truncated_text})
            
        return final_messages
