'''
Implements an NLP chatbot with custom context management.

Refactored to separate the core ChatBot logic from the History Management logic.

Files Overview:
    - chatbot.py: Main entry point class `ChatBot` handling Flask routes and model generation.
    - history_manager.py: Contains `ConversationContext` (state) and `HistoryManager` (strategy) classes.
    - chatbot_old.py: The legacy monolithic implementation (kept for reference/backup).

Classes:
    - ChatBot: The primary controller. It accepts a `HistoryManager` strategy but currently defaults 
      to `TokenTruncationManager`.

Usage:
    The `ChatBot` class initializes a `ConversationContext` to hold state. 
    It delegates history merging to `self.history_manager.process(...)`.

Author: Jaideep Murkute
Date: 2025-12-27
'''

import gc
from datetime import datetime
import os
import torch
from flask import render_template, request, jsonify

from CFG import Config
from model_singleton import ModelSingleton
from utils import *

# Import the new Strategy Pattern components
from history_manager import ConversationContext, TokenTruncationManager

class ChatBot:
    """
    The main class for ChatBot application.
    
    Responsibilities:
        1.  Initialize Model and Tokenizer (via Singleton).
        2.  Load System Prompt.
        3.  Manage Web Routes (Flask).
        4.  Generate Responses (delegating history logic to HistoryManager).
    
    Attributes:
        cfg (dict): Configuration settings.
        app (Flask): Flask app instance.
        context (ConversationContext): Holds the conversation state (history, logs, system prompt).
        history_manager (HistoryManager): The strategy for managing conversation history.
    """
    def __init__(self, cfg: dict, app) -> None:
        self.cfg = cfg
        self.app = app
        self._setup_routes() 
        
        self.model_singleton = ModelSingleton(self.cfg)
        self.model = self.model_singleton.model
        self.tokenizer = self.model_singleton.tokenizer
        print("Model and tokenizer loaded successfully !!!")
        
        # Initialize Context
        self.context = ConversationContext()
        
        # Initialize Strategy (Defaulting to TokenTruncationManager as per original logic)
        self.history_manager = TokenTruncationManager()

        self._load_system_prompt()
        self._init_session_state()
        self._setup_gen_kwargs()

    def _setup_gen_kwargs(self):
        """Prepares the generation arguments from the config."""
        self.gen_kwargs = {
            'max_length': self.cfg.get('max_len', 1000),
            'pad_token_id': self.tokenizer.eos_token_id,
            'num_beams': self.cfg.get('num_beams', 1),
            'temperature': self.cfg.get('temperature', 1.0),
            'top_k': self.cfg.get('top_k', 50),
            'top_p': self.cfg.get('top_p', 1.0),
            'repetition_penalty': self.cfg.get('repetition_penalty', 1.0),
            'no_repeat_ngram_size': self.cfg.get('no_repeat_ngram_size', 0),
            'num_return_sequences': self.cfg.get('num_return_sequences', 1)
        }
        
        # Automatically enable sampling if not using beam search and temperature is set
        if self.gen_kwargs['num_beams'] == 1 and self.gen_kwargs['temperature'] != 1.0:
            self.gen_kwargs['do_sample'] = True
        if self.gen_kwargs['num_beams'] > 1:
            self.gen_kwargs['early_stopping'] = True
            
        print(f"Generation kwargs setup: {self.gen_kwargs}")

    def _load_system_prompt(self):
        """Loads and tokenizes the system prompt."""
        system_prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                raw_text = f.read().strip()
                self.system_prompt_text = raw_text
                system_prompt_text = raw_text + "\n" 
            
            sys_enc = self.tokenizer.encode_plus(system_prompt_text, return_tensors='pt', padding=False, truncation=False)
            
            # Store system prompt in context
            self.context.system_ids = sys_enc['input_ids']
            self.context.system_att_mask = sys_enc['attention_mask']
            print(f"System prompt loaded. Length: {len(self.context.system_ids[0])} tokens.")
        except Exception as e:
            print(f"Warning: Could not load system_prompt.txt: {e}")
            # Defaults
            self.system_prompt_text = "You are a helpful assistant."
            self.context.system_ids = torch.tensor([[]])
            self.context.system_att_mask = torch.tensor([[]])

    def _init_session_state(self):
        """Initializes the session logging in context."""
        # Separate metadata from conversation logs
        self.context.session_metadata = {
            'session_id': self.cfg['session_id'], 
            'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }
        self.context.logs = [] # Pure list of conversation turns
        
        # Reset tensors in context
        self.context.history_ids = torch.tensor([])
        self.context.attention_mask = torch.tensor([])

    def generate_response(self, user_input: str) -> None:
        '''
        Generates the model's response for the given user input.
        
        Process:
        1. Tokenize user input.
        2. Delegate history merging to self.history_manager.process().
        3. Prepend system prompt (if exists).
        4. Generate via model.
        5. Update context with response.
        '''
        # Determine if we should use Chat Templates (for modern Instruct models) \
        # or Legacy formatting
        use_chat_template = "Instruct" in self.cfg['model_name'] or "Chat" in self.cfg['model_name'] or \
                            "Qwen" in self.cfg['model_name']
        
        if use_chat_template:
            # --- MODERN INSTRUCT PIPELINE ---
            # Use HistoryManager to construct the optimized message list (System + Truncated History + User)
            system_prompt = getattr(self, 'system_prompt_text', "You are a helpful assistant.")
            
            # This call encapsulates all the truncation logic (max_tot_input_prop, etc.) for lists
            final_messages = self.history_manager.process_messages(
                messages=self.context.logs, 
                system_prompt=system_prompt, 
                user_input=user_input, 
                config=self.cfg, 
                tokenizer=self.tokenizer
            )

            # Apply Template
            # This returns tensor of ids
            print("final_messages: ", final_messages)
            final_input_ids = self.tokenizer.apply_chat_template(
                final_messages, 
                tokenize=True, 
                add_generation_prompt=True,
                return_tensors="pt"
            )
            final_att_mask = torch.ones_like(final_input_ids) # Simple mask for freshly generated sequence
            
        else:
            # --- LEGACY PIPELINE (DialoGPT, finetuned dialoGPT) ---
            user_ip_enc = self.tokenizer.encode_plus(user_input + self.tokenizer.eos_token, 
                            return_tensors='pt', padding=True, truncation=True)
            
            # DELEGATION: Let the strategy handle history merging/truncation
            self.history_manager.process(self.context, user_ip_enc, self.cfg)
            
            # PREPEND SYSTEM PROMPT FOR GENERATION (Transient Injection)
            if self.context.system_ids.numel() > 0:
                 # Ensure system ids are on the same device as bot_ip_ids (history_ids)
                device = self.context.history_ids.device
                sys_ids = self.context.system_ids.to(device)
                sys_mask = self.context.system_att_mask.to(device)
                
                final_input_ids = torch.cat([sys_ids, self.context.history_ids], dim=-1)
                final_att_mask = torch.cat([sys_mask, self.context.attention_mask], dim=-1)
            else:
                final_input_ids = self.context.history_ids
                final_att_mask = self.context.attention_mask

        # print("final_input_ids: ", final_input_ids)
        # print("final_input_ids.shape: ", final_input_ids.shape)
        # print("self.cfg['max_len]: ", self.cfg['max_len'])
        
        print(f"Generating with args: {self.gen_kwargs}")

        # For causal models; model.generate() returns full sequence: 
        # (prompt / input ids) + (newly generated ids).
        model_op_ids = self.model.generate(final_input_ids, attention_mask=final_att_mask, 
                                            **self.gen_kwargs)
        
        # print("model_op_ids: ", model_op_ids)
        # Decode response: slice off the input (system + history + current)
        # We slice from final_input_ids.shape[-1]
        response_text = self.tokenizer.decode(model_op_ids[:, final_input_ids.shape[-1]:][0], \
                                skip_special_tokens=True)
        print("response_text: ", response_text)

        # Update Logs
        # Now logs are purely conversation turns, no metadata mixing.
        self.context.logs.append({"user": user_input, "model": response_text})
        
        # Update Context with Response (Append to history - Legacy Support) 
        # We do this for BOTH pipelines so that history state is preserved if we switch models or save session
        resp_enc = self.tokenizer.encode_plus(response_text + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        # For legacy pipeline, this appending happens naturally. 
        # For instruct pipeline, we need to manually update the tracking tensors.
        if use_chat_template:
            # For instruct, we might just want to append the interaction tokens to history_ids 
            # loosely to keep the buffer alive, or rely solely on logs.
            # To be safe and compatible with history_manager.process input expectation:
            curr_turn_ids = self.tokenizer.encode_plus(user_input + self.tokenizer.eos_token + response_text + self.tokenizer.eos_token,
                                                       return_tensors='pt', padding=True, truncation=True)
            self.context.history_ids = torch.cat([self.context.history_ids, curr_turn_ids['input_ids']], dim=-1)
            self.context.attention_mask = torch.cat([self.context.attention_mask, curr_turn_ids['attention_mask']], dim=-1)
        else:
            self.context.history_ids = torch.cat([self.context.history_ids, resp_enc['input_ids']], dim=-1)
            self.context.attention_mask = torch.cat([self.context.attention_mask, resp_enc['attention_mask']], dim=-1)
        
        # Cache response for the route handler
        self.last_response = response_text
        print()

    def _setup_routes(self) -> None:
        self.app.route('/')(self._index)
        self.app.route('/model_info', methods=['GET'])(self._model_info)
        self.app.route('/chat', methods=['POST'])(self._chat)
        self.app.route('/close_chat', methods=['POST'])(self._close_chat)
        self.app.route('/new_session', methods=['POST'])(self._new_session)
    
    def _index(self):
        return render_template('index.html')
    
    def _model_info(self):
        return jsonify(model_name=self.cfg['model_name'])

    def _chat(self):
        user_input = request.form['user_input']
        self.generate_response(user_input)
        return jsonify(response=self.last_response)

    def _close_chat(self):
        print("Chat closed by user.")
        # Bundle metadata + logs for saving
        final_log_dump = [self.context.session_metadata] + self.context.logs
        save_conversations(self.cfg, final_log_dump)
        
        # basic cleanup
        self.context = None
        gc.collect()
        
        return jsonify(message="Chat closed successfully.")

    def _new_session(self):
        '''
        Starts a new chatbot session.
        '''
        # save the current conversation logs
        if self.context and self.context.logs:
             final_log_dump = [self.context.session_metadata] + self.context.logs
             save_conversations(self.cfg, final_log_dump)
        print("Cleaning data for session ID: ", self.cfg['session_id'])

        # reset the config and create new directories with new session id
        config = Config()
        self.cfg = config.config
        self.cfg = create_dirs_paths(self.cfg)
        save_config(self.cfg)
        
        # Re-initialize context state for new session
        self._init_session_state()
        self._setup_gen_kwargs()
        self._load_system_prompt() # Reload text file to apply any changes
        
        print(f"New chatbot session initialized successfully !!!")
        print("New session ID: ", self.cfg['session_id'])
        
        return jsonify(success=True)
