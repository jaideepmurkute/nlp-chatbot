"""
NLP Chatbot Application

This application implements an NLP chatbot with custom context management to allow for improved conversation 
quality with shorter context size models.

Supports custom behaviours based on the configuration settings in CFG.py.

Supports three modes of context management:
    1] Vanilla: The chatbot uses a fixed size context window for the conversation history.
    2] prop_slice: For the long history and inputs; the chatbot dynamically adjusts the context 
                and input size based on the proportions set by the user.
    2] summarize_prop_slice: The chatbot dynamically adjusts the context and input size based by 
            first generating the summary of long history or the input and then applying proportional 
            slicing to the inputs.

Author: Jaideep Murkute
Date: 2024-07-29
Version: 1.0

"""


from datetime import datetime
import sys

from flask import Flask, render_template, request, jsonify
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

    Methods
    -------
    __init__(self, cfg, app)
        Initializes the ChatBot with the given configuration and app.
    """
    def __init__(self, cfg, app) -> None:
        self.cfg = cfg
        self.app = app
        
        self.model_singleton = ModelSingleton(self.cfg)
        self.model = self.model_singleton.model
        self.tokenizer = self.model_singleton.tokenizer
        print("Model and tokenizer loaded successfully !!!")
        
        self.setup_routes()
        
        self.convos = [{'session_id': cfg['session_id'], 
                    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        
        self.bot_ip_ids = torch.tensor([])
        self.bot_att_mask = torch.tensor([])
        
    
    def find_hist_prop(self, step_size=0.1):
            """
            Finds the maximum proportion for the amount of historical input to use in the total input.
            Based on current history state and input and proportion parameters in the configuration.

            Parameters:
            - step_size (float): The step size to increment the historical input proportion. Default is 0.1.

            Returns:
            - hist_prop_to_use (float): The historical input proportion to use.
            """
            
            num_steps = int((self.cfg['max_hist_input_prop'] - self.cfg['min_hist_input_prop']) / step_size) + 1
            hist_prop_lst = [self.cfg['min_hist_input_prop'] + (i*step_size) for i in range(num_steps)]

            hist_prop_to_use = None
            for hist_prop in hist_prop_lst:
                new_hist_len = int(hist_prop * self.curr_hist_len)
                if new_hist_len + self.curr_ip_len <= self.max_tot_ip_len:
                    hist_prop_to_use = hist_prop
                    break

            if hist_prop_to_use is None: hist_prop_to_use = self.cfg['min_hist_input_prop']

            return hist_prop_to_use
        
    def merge_history(self):
        '''
        Merges the current user input with the historical conversation context.
        
        If the combined input and history exceed the maximum allowed size for the total model input,
        it finds the maximum proportion of history and input that can fit and keeps that portion 
        to feed to the model.
        
        Parameters:
        - None
        
        Returns:
        - None
        '''
        if len(self.bot_ip_ids) == 0:
            self.bot_ip_ids = self.user_ip_enc['input_ids']
            self.bot_att_mask = self.user_ip_enc['attention_mask']
        else:
            self.curr_ip_len = self.user_ip_enc['input_ids'].shape[-1]
            self.curr_hist_len = self.bot_ip_ids.shape[-1]
            self.curr_tot_ip_len = self.curr_ip_len + self.curr_hist_len
            
            if self.curr_tot_ip_len > int(self.cfg['max_len'] * self.cfg['max_tot_input_prop']):
                self.max_tot_ip_len = int(self.cfg['max_len'] * self.cfg['max_tot_input_prop'])

                hist_prop_to_use = self.find_hist_prop()
                new_hist_len = int(hist_prop_to_use * self.curr_hist_len)
                
                self.bot_ip_ids = self.bot_ip_ids[:, -new_hist_len:]
                self.bot_att_mask = self.bot_att_mask[:, -new_hist_len:]
                
                if new_hist_len + self.curr_ip_len > self.max_tot_ip_len:
                    new_ip_len = self.max_tot_ip_len - new_hist_len
                    self.user_ip_enc['input_ids'] = self.user_ip_enc['input_ids'][:, :new_ip_len]
                    self.user_ip_enc['attention_mask'] = self.user_ip_enc['attention_mask'][:, :new_ip_len]
            
            self.bot_ip_ids = torch.cat([self.bot_ip_ids, self.user_ip_enc['input_ids']], dim=-1) 
            self.bot_att_mask = torch.cat([self.bot_att_mask, self.user_ip_enc['attention_mask']], dim=-1)
        
    def generate_response(self, user_input):
        '''
        Function handles the core chatbot logic:
        1] Encodes the user input
        2] Calls the merge_history method to merge the user input with the historical context
        3] Generates the model's response for the given user input
        4] Response is saved in the response attribute and also in the convos attribute
        
        Parameters:
        - user_input (str): The input text from the user.
        
        Returns:
        - None 
        '''
        self.user_ip_enc = self.tokenizer.encode_plus(user_input + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        self.merge_history()
        
        model_op_ids = self.model.generate(self.bot_ip_ids, attention_mask=self.bot_att_mask, 
                        max_length=cfg['max_len'], pad_token_id=self.tokenizer.eos_token_id)
            
        self.response = self.tokenizer.decode(model_op_ids[:, self.bot_ip_ids.shape[-1]:][0], \
                                skip_special_tokens=True)
        
        self.convos.append({"user": user_input})
        self.convos[-1]["model"] = self.response
        
        resp_enc = self.tokenizer.encode_plus(self.response + self.tokenizer.eos_token, 
                        return_tensors='pt', padding=True, truncation=True)
        
        self.bot_ip_ids = torch.cat([self.bot_ip_ids, resp_enc['input_ids']], dim=-1)
        self.bot_att_mask = torch.cat([self.bot_att_mask, resp_enc['attention_mask']], dim=-1)

    def setup_routes(self):
        '''
        Function encapsulates the Flask web application routes handlers for the ChatBot application.
        '''
        @self.app.route('/')
        def index(self):
            return render_template('index.html')

        @self.app.route('/chat', methods=['POST'])
        def chat(self):
            user_input = request.form['user_input']
            self.generate_response(user_input)
            return jsonify(response=self.response)
        
        @self.app.route('/close_chat', methods=['POST'])
        def close_chat(self):
            print("Chat closed by user.")
            # Perform any cleanup if necessary
            save_conversations(self.cfg, self.convos)
            return jsonify(message="Chat closed successfully.")
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info(self):
            return jsonify(model_name=self.cfg['model_name'])

        @self.app.route('/new_session', methods=['POST'])
        def new_session(self):
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
            self.cfg = create_dirs_paths(cfg)
            save_config(self.cfg)
            
            # reset conversation tracking variables
            self.convos = [{'session_id': cfg['session_id'], 
                    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
            self.bot_ip_ids = torch.tensor([])
            self.bot_att_mask = torch.tensor([])
            
            print(f"New chatbot session initialized successfully !!!")
            print("New session ID: ", self.cfg['session_id'])
            
            return jsonify(success=True)


app = Flask(__name__)


# @app.route('/')
# def index():
#     return render_template('index.html')

'''
Main function to run the ChatBot application.
'''


if __name__ == "__main__":
    config = Config()
    cfg = config.config

    cfg = create_dirs_paths(cfg)
    save_config(cfg)

    chatbot = ChatBot(cfg, app)

    try:
        # app.run(debug=True)
        # app.run(host='0.0.0.0')
        app.run(host="0.0.0.0", port=5100)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving conversation logs...")
        save_conversations(cfg, chatbot.convos)
        print("Conversation logs saved. Exiting chatbot !!!")

    