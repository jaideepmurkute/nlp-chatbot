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

from .summarizer import Summarizer


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
    def process(
        self, context: ConversationContext, user_input_enc: dict, config: dict
    ) -> None:
        """
        Modifies the context in-place based on the user input and the specific strategy logic.

        Args:
            context (ConversationContext): The current conversation state.
            user_input_enc (dict): The tokenized user input (must contain 'input_ids' and 'attention_mask').
            config (dict): The configuration dictionary containing limits like 'max_len', 'max_tot_input_prop', etc.
        """
        pass

    @abstractmethod
    def process_messages(
        self,
        messages: List[Dict],
        system_prompt: str,
        user_input: str,
        config: dict,
        tokenizer,
    ) -> List[Dict]:
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

    As of now, supports two Data Pipelines:
    A. `process`: Legacy - Handles raw `torch.Tensor` concatenation (e.g., for DialoGPT).
        Can delete later and end support for older model families.
    B. `process_messages`: Modern - Handles structured `List[Dict]` pruning
        (e.g. for Qwen/Llama Chat Templates).
    """

    def process(
        self, context: ConversationContext, user_input_enc: dict, config: dict
    ) -> None:
        """
        LEGACY PIPELINE: Merges user input into context tensors with intelligent truncation.

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

        curr_ip_len = user_input_enc["input_ids"].shape[-1]

        # Handle initial case
        if context.history_ids.numel() == 0:
            context.history_ids = user_input_enc["input_ids"]
            context.attention_mask = user_input_enc["attention_mask"]
            return

        curr_hist_len = context.history_ids.shape[-1]
        curr_tot_ip_len = curr_ip_len + curr_hist_len

        # DEDUCT SYSTEM PROMPT LENGTH TO RESERVE SPACE
        system_len = (
            context.system_ids.shape[-1] if context.system_ids.numel() > 0 else 0
        )
        max_permissible_ip_tokens = (
            int(config["max_len"] * config["max_tot_input_prop"]) - system_len
        )

        # If total length falls within limits, no truncation needed
        if curr_tot_ip_len <= max_permissible_ip_tokens:
            # Append directly
            context.history_ids = torch.cat(
                [context.history_ids, user_input_enc["input_ids"]], dim=-1
            )
            context.attention_mask = torch.cat(
                [context.attention_mask, user_input_enc["attention_mask"]], dim=-1
            )
        else:
            # Calculate minimum history to sustain (e.g. 10% of current history)
            min_hist_needed = int(curr_hist_len * config["min_hist_input_prop"])

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
                user_input_enc["input_ids"] = user_input_enc["input_ids"][
                    :, :new_ip_len
                ]
                user_input_enc["attention_mask"] = user_input_enc["attention_mask"][
                    :, :new_ip_len
                ]

            # Merge
            context.history_ids = torch.cat(
                [context.history_ids, user_input_enc["input_ids"]], dim=-1
            )
            context.attention_mask = torch.cat(
                [context.attention_mask, user_input_enc["attention_mask"]], dim=-1
            )

    def process_messages(
        self,
        messages: List[Dict],
        system_prompt: str,
        user_input: str,
        config: dict,
        tokenizer,
    ) -> List[Dict]:
        """
        MODERN PIPELINE: Implementation of the Token Truncation logic for modern chat templates.
        """
        final_messages = []

        # Compute lengths of system and user inputs, ignoring special tokens for clarity
        sys_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
        user_ids = tokenizer.encode(user_input, add_special_tokens=False)

        sys_len = len(sys_ids)
        user_len = len(user_ids)

        # Determine limits - how many tokens can be used for history and input
        max_len = config.get("max_len", 2048)
        max_tot_input_prop = config.get("max_tot_input_prop", 0.85)
        limit = int(max_len * max_tot_input_prop)

        # Fit components
        # Calculate available space for history
        available_curr_len = sys_len + user_len

        if available_curr_len > limit:
            # Edge Case: System + User exceeds limit.
            # We'd truncate user input to fit.
            avail_for_user = limit - sys_len
            if avail_for_user < 0:
                avail_for_user = 0

            # Decode back to string to ensure valid text
            user_input = tokenizer.decode(user_ids[:avail_for_user])

            # No history possible to be included
            history_msgs = []
        else:
            # Iterate backwards adding history until full
            available_for_history = limit - available_curr_len
            history_msgs = []
            curr_hist_len = 0

            for msg in reversed(messages):
                msg_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
                msg_len = len(msg_ids)

                if curr_hist_len + msg_len <= available_for_history:
                    history_msgs.insert(0, msg)
                    curr_hist_len += msg_len
                else:
                    # Stop if we can't fit more
                    break

        # Construct Final List
        final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(history_msgs)
        final_messages.append({"role": "user", "content": user_input})

        return final_messages


class SummarizationHistoryManager(HistoryManager):
    """
    Concrete Strategy implementing Summarization-based History Management.
    """

    def __init__(self, summarizer_model=None, summarizer_tokenizer=None, device="cpu"):
        self.current_summary = ""
        self.summarizer = None

        if summarizer_model and summarizer_tokenizer:
            self.summarizer = Summarizer(summarizer_model, summarizer_tokenizer, device)
        else:
            print(
                "Warning: SummarizationHistoryManager initialized without model/tokenizer. \
                Summarization will fail."
            )

    def process(
        self, context: ConversationContext, user_input_enc: dict, config: dict
    ) -> None:
        """
        Legacy Tensor Pipeline - Not implemented for Summarization currently.
        Could fallback to truncation or raise warning.
        """
        print(
            "Warning: Summarization strategy not implemented for legacy tensor pipeline. \
                Using fallback."
        )
        context.history_ids = torch.cat(
            [context.history_ids, user_input_enc["input_ids"]], dim=-1
        )
        context.attention_mask = torch.cat(
            [context.attention_mask, user_input_enc["attention_mask"]], dim=-1
        )

    def process_messages(
        self,
        messages: List[Dict],
        system_prompt: str,
        user_input: str,
        config: dict,
        tokenizer,
    ) -> List[Dict]:
        """
        Constructs context using Summary + Buffer.

        CRITICAL: This method also MUTATES the 'messages' list (by reference) if summarization occurs!
        The 'messages' list passed here is 'self.context.logs' from ChatBot.
        """
        final_messages = []

        # Check for buffer overflow & summarize
        max_buffer = config.get("max_buffer_msgs", 5)

        if len(messages) > max_buffer:
            # Separate overflow vs buffer
            overflow_msgs = messages[:-max_buffer]

            # Update Summary
            if self.summarizer:
                # This might take time (LLM call)
                self.current_summary = self.summarizer.summarize(
                    self.current_summary, overflow_msgs
                )
                print(
                    f"[SummarizationHistoryManager] Updated Summary: {self.current_summary[:50]}..."
                )

            # Delete old messages from memory
            del messages[:-max_buffer]

        # Add system prompt
        final_messages.append({"role": "system", "content": system_prompt})

        # Add summary, if exists
        if self.current_summary:
            # Inject Summary into System Prompt or as a separate System block
            # Appending to the system content is usually robust.
            final_messages[0]["content"] += (
                f"\n\nPrevious Conversation Summary:\n{self.current_summary}"
            )

        # Add remaining messages
        final_messages.extend(messages)

        # Add user input
        final_messages.append({"role": "user", "content": user_input})

        return final_messages
