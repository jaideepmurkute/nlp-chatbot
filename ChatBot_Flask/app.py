
from datetime import datetime

from flask import Flask, render_template, request, jsonify
import torch

from CFG import Config
from utils import *
from model_singleton import ModelSingleton


class ChatBot:
    def __init__(self, cfg, app) -> None:
        self.cfg = cfg
        self.app = app
        
        self.model_singleton = ModelSingleton(self.cfg)
        self.model = self.model_singleton.model
        self.tokenizer = self.model_singleton.tokenizer

        self.setup_routes()
        
        self.convos = [{'session_id': cfg['session_id'], 
                    'datetime': datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        
        self.bot_ip_ids = torch.tensor([])
        self.bot_att_mask = torch.tensor([])
        
    
    def find_hist_prop(self, step_size=0.1):
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
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/chat', methods=['POST'])
        def chat():
            user_input = request.form['user_input']
            self.generate_response(user_input)
            return jsonify(response=self.response)
        
        @self.app.route('/close_chat', methods=['POST'])
        def close_chat():
            print("Chat closed by user.")
            # Perform any cleanup if necessary
            save_conversations(self.cfg, self.convos)
            return jsonify(message="Chat closed successfully.")
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            return jsonify(model_name=self.cfg['model_name'])

        @app.route('/new_session', methods=['POST'])
        def new_session():
            # Logic to start a new session
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

        

if __name__ == "__main__":
    config = Config()
    cfg = config.config
    
    cfg = create_dirs_paths(cfg)
    save_config(cfg)
    
    app = Flask(__name__)
    chatbot = ChatBot(cfg, app)
    
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving conversation logs...")
        save_conversations(cfg, chatbot.convos)
        print("Conversation logs saved. Exiting chatbot !!!")
    
    