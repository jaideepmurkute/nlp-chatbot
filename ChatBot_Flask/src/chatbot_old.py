'''
Implements an NLP chatbot with custom context management to allow for improved conversation 
quality with shorter context size models.

Supports custom behaviours based on the configuration settings in CFG.py.

Supports three modes of context management:
    1] Vanilla: The chatbot uses a fixed size context window for the conversation history.
    2] prop_slice: For the long history and inputs; the chatbot dynamically adjusts the context 
                and input size based on the proportions set by the user.
    3] summarize_prop_slice: The chatbot dynamically adjusts the context and input size based by 
            first generating the summary of long history or the input and then applying proportional 
            slicing to the inputs.

Author: Jaideep Murkute
Date: 2025-12-26 
Version: 1.2 (System Prompt Support)

'''
import gc
from datetime import datetime
import os

from flask import render_template, request, jsonify
import torch

from CFG import Config
from model_singleton import ModelSingleton
from utils import *


class ChatBot:
    """
    The main class for ChatBot application.
        1] Implements the Core chatbot logic.
        2] Implements custom history/context management for chatbot.
        3] Sets up the Flask routes for chatbot.

    Attributes
    ----------
    cfg : dict
        Configuration settings for the ChatBot.
    app : Flask
        The Flask application instance.

    Variables info:
        bot_ip_ids : input token ids passed to the chatbot.
        bot_att_mask : attention mask passed to the chatbot. Same size as bot_ip_ids.

        max_permissible_ip_tokens: Max how many tokens long can total input be?
            In model architecture, input tokens = output tokens.
            So, input size + output size <= context size
            We must reserve some space for output tokens.
            We set `max_tot_input_prop` to set max size for input; rest is for output
    """
    def __init__(self, cfg: dict, app) -> None:
        self.cfg = cfg
        self.app = app
        self._setup_routes() # setup the Flask action routes
        
        self.model_singleton = ModelSingleton(self.cfg)
        self.model = self.model_singleton.model
        self.tokenizer = self.model_singleton.tokenizer
        print("Model and tokenizer loaded successfully !!!")
        
        # Load and tokenize system prompt
        system_prompt_path = os.path.join(os.path.dirname(__file__), 'system_prompt.txt')
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt_text = f.read().strip() + "\n" # Add newline separator
            
            sys_enc = self.tokenizer.encode_plus(system_prompt_text, return_tensors='pt', 
                        padding=False, truncation=False)
            self.system_ip_ids = sys_enc['input_ids']
            self.system_att_mask = sys_enc['attention_mask']
            print(f"System prompt loaded. Length: {len(self.system_ip_ids[0])} tokens.")
        except Exception as e:
            print(f"Warning: Could not load system_prompt.txt: {e}")
            self.system_ip_ids = torch.tensor([[]])
            self.system_att_mask = torch.tensor([[]])

        self.convos = [{'session_id': cfg['session_id'], 
                    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        
        self.bot_ip_ids = torch.tensor([])
        self.bot_att_mask = torch.tensor([])
        
    
    def _merge_history(self) -> None:
        '''
        Merges the current user input with the historical conversation context.
        'bot_ip_ids' actually only builds up to: 
            max context size minus the prompt size minus the reserved space for response.

        Logic:
        1. Calculate the maximum allowed tokens for total input (history + current input).
           *Deduct* the system prompt length from this limit to reserve space.
        2. If current input + history fits, use everything.
        3. If it doesn't fit:
            a. Determine minimum required history (10% of current history).
            b. Calculate space available for history after accommodating the full current input.
            c. If space available >= minimum required history:
                - Keep as much history as possible (max_allowed - input_len).
            d. If space available < minimum required history (Input is too large):
                - Keep minimum required history.
                - Truncate current input to fit in the remaining space.
        '''
        if len(self.bot_ip_ids) == 0:
            # first message - no history
            self.bot_ip_ids = self.user_ip_enc['input_ids']
            self.bot_att_mask = self.user_ip_enc['attention_mask']
        else:
            self.curr_ip_len = self.user_ip_enc['input_ids'].shape[-1]
            self.curr_hist_len = self.bot_ip_ids.shape[-1]
            self.curr_tot_ip_len = self.curr_ip_len + self.curr_hist_len
            
            # We reserve minimum portion of the context size for response - by 
            # setting maximum limit on history + current input
            
            # DEDUCT SYSTEM PROMPT LENGTH TO RESERVE SPACE
            system_len = self.system_ip_ids.shape[-1] if self.system_ip_ids.numel() > 0 else 0
            max_permissible_ip_tokens = int(self.cfg['max_len'] * self.cfg['max_tot_input_prop']) - system_len
            
            # If total length falls within limits, no truncation needed
            if self.curr_tot_ip_len <= max_permissible_ip_tokens:
                self.max_tot_ip_len = max_permissible_ip_tokens 
                # Append directly
                self.bot_ip_ids = torch.cat([self.bot_ip_ids, self.user_ip_enc['input_ids']], dim=-1) 
                self.bot_att_mask = torch.cat([self.bot_att_mask, self.user_ip_enc['attention_mask']], dim=-1)
            else:
                self.max_tot_ip_len = max_permissible_ip_tokens
                
                # Calculate minimum history to sustain (e.g. 10% of current history)
                min_hist_needed = int(self.curr_hist_len * self.cfg['min_hist_input_prop'])
                
                # Calculate how much space is left for history if we keep the full input
                space_for_history = max_permissible_ip_tokens - self.curr_ip_len
                
                new_hist_len = 0
                new_ip_len = self.curr_ip_len
                
                if space_for_history >= min_hist_needed:
                    # Scenario: We can fit the full input and at least the minimum history.
                    # We fill the remaining space with history.
                    new_hist_len = space_for_history
                    # Input remains full size (new_ip_len is already self.curr_ip_len)
                else:
                    # Scenario: Input is so large that we can't even fit minimum history.
                    # Priority: Maintain minimum history, truncate input.
                    new_hist_len = min_hist_needed
                    
                    # Remaining space goes to input
                    new_ip_len = max_permissible_ip_tokens - new_hist_len
                
                # Apply Truncation to History
                # We keep the *latest* history (last new_hist_len tokens)
                if new_hist_len > 0:
                    self.bot_ip_ids = self.bot_ip_ids[:, -new_hist_len:]
                    self.bot_att_mask = self.bot_att_mask[:, -new_hist_len:]
                else:
                    # Corner case: if min_hist_needed is 0 and space_for_history is 0 (unlikely but safe to handle)
                    self.bot_ip_ids = torch.tensor([[]], device=self.bot_ip_ids.device)
                    self.bot_att_mask = torch.tensor([[]], device=self.bot_att_mask.device)

                # Apply Truncation to Input if needed
                if new_ip_len < self.curr_ip_len:
                    # We keep the start of the input as per original code behavior ([:, :len])
                    self.user_ip_enc['input_ids'] = self.user_ip_enc['input_ids'][:, :new_ip_len]
                    self.user_ip_enc['attention_mask'] = self.user_ip_enc['attention_mask'][:, :new_ip_len]
                
                # Merge
                self.bot_ip_ids = torch.cat([self.bot_ip_ids, self.user_ip_enc['input_ids']], dim=-1) 
                self.bot_att_mask = torch.cat([self.bot_att_mask, self.user_ip_enc['attention_mask']], dim=-1)
    

    def generate_response(self, user_input: str) -> None:
        '''
        Function handles the core chatbot logic:
        1] Encodes the user input
        2] Calls the _merge_history() method to merge the user input with the historical context
        3] Generates the model's response for the given user input
        4] Response is saved in the response attribute and also in the convos attribute
        
        NOTE: system prompt is not included in the history. So, doesnt get redundant or sliced away
            by the _merge_history() method.

        Parameters:
        - user_input (str): The input text from the user.
        '''
        self.user_ip_enc = self.tokenizer.encode_plus(user_input + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        # Merge current prompt with historical context; ensure some minimum space of the context 
        # window is left for the output.
        self._merge_history()
        
        # PREPEND SYSTEM PROMPT FOR GENERATION (Transient Injection)
        if self.system_ip_ids.numel() > 0:
             # Ensure system ids are on the same device as bot_ip_ids
            sys_ids = self.system_ip_ids.to(self.bot_ip_ids.device)
            sys_mask = self.system_att_mask.to(self.bot_att_mask.device)
            
            final_input_ids = torch.cat([sys_ids, self.bot_ip_ids], dim=-1)
            final_att_mask = torch.cat([sys_mask, self.bot_att_mask], dim=-1)
        else:
            final_input_ids = self.bot_ip_ids
            final_att_mask = self.bot_att_mask

        model_op_ids = self.model.generate(final_input_ids, attention_mask=final_att_mask, 
                        max_length=self.cfg['max_len'], pad_token_id=self.tokenizer.eos_token_id)
            
        # Decode response: slice off the input (system + history + current)
        # We slice from final_input_ids.shape[-1]
        self.response = self.tokenizer.decode(model_op_ids[:, final_input_ids.shape[-1]:][0], \
                                skip_special_tokens=True)
        
        self.convos.append({"user": user_input})
        self.convos[-1]["model"] = self.response
        
        resp_enc = self.tokenizer.encode_plus(self.response + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        self.bot_ip_ids = torch.cat([self.bot_ip_ids, resp_enc['input_ids']], dim=-1)
        self.bot_att_mask = torch.cat([self.bot_att_mask, resp_enc['attention_mask']], dim=-1)

    def _setup_routes(self) -> None:
        '''
        Function encapsulates the Flask web application routes handlers for the ChatBot application.
        '''
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
        return jsonify(response=self.response)

    def _close_chat(self):
        print("Chat closed by user.")
        save_conversations(self.cfg, self.convos)
        
        # basic cleanup
        self.bot_ip_ids = None
        self.bot_att_mask = None
        self.convos = None
        gc.collect()
        
        return jsonify(message="Chat closed successfully.")

    
    def _new_session(self):
        '''
        Starts a new chatbot session by: 
            1] Saving past conversation logs.
            2] Creates new directories with a new session id.
            3] Resets the config and class states. 
        '''
        # save the current conversation logs
        save_conversations(self.cfg, self.convos)
        print("Cleaning data for session ID: ", self.cfg['session_id'])

        # reset the config and create new directories with new session id
        config = Config()
        self.cfg = config.config
        self.cfg = create_dirs_paths(self.cfg)
        save_config(self.cfg)
        
        # reset conversation tracking variables
        self.convos = [{'session_id': self.cfg['session_id'], 
                'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        self.bot_ip_ids = torch.tensor([])
        self.bot_att_mask = torch.tensor([])
        
        print(f"New chatbot session initialized successfully !!!")
        print("New session ID: ", self.cfg['session_id'])
        
        return jsonify(success=True)
