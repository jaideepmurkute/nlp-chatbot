
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
Version: 1.1 (Logic Correction)

'''
import gc
from datetime import datetime

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
        
        self.convos = [{'session_id': cfg['session_id'], 
                    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        
        self.bot_ip_ids = torch.tensor([])
        self.bot_att_mask = torch.tensor([])
        
    
    def _merge_history(self) -> None:
        '''
        Merges the current user input with the historical conversation context.
        
        Logic:
        1. Calculate the maximum allowed tokens for total input (history + current input).
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
            max_permissible_ip_tokens = int(self.cfg['max_len'] * self.cfg['max_tot_input_prop'])
            
            # If total length falls within limits, no truncation needed
            if self.curr_tot_ip_len <= max_permissible_ip_tokens:
                self.max_tot_ip_len = max_permissible_ip_tokens # Just storing it as per original class structure
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
        2] Calls the merge_history method to merge the user input with the historical context
        3] Generates the model's response for the given user input
        4] Response is saved in the response attribute and also in the convos attribute
        
        Parameters:
        - user_input (str): The input text from the user.
        '''
        self.user_ip_enc = self.tokenizer.encode_plus(user_input + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        # Merge current prompt with historical context; ensure some minimum space of the context 
        # window is left for the output.
        self._merge_history()
        
        model_op_ids = self.model.generate(self.bot_ip_ids, attention_mask=self.bot_att_mask, 
                        max_length=self.cfg['max_len'], pad_token_id=self.tokenizer.eos_token_id)
            
        self.response = self.tokenizer.decode(model_op_ids[:, self.bot_ip_ids.shape[-1]:][0], \
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
