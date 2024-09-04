
'''
Utility functions for the ChatBot.

__author__ =  ''
__email__ = ''
__version__ = ''
'''

import os
import sys
import random
import string

import json
import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(cfg):
    '''
    Get the model and tokenizer from the checkpoint directory if it exists, else download from HuggingFace.
    Args:
        cfg (dict): Configuration dictionary
    Returns:
        model (AutoModelForCausalLM): Model instance
        tokenizer (AutoTokenizer): Tokenizer instance
    '''
    checkpoint_dir = os.path.join(cfg["model_store_dir"], cfg["model_name"])
    if os.path.exists(cfg["model_store_dir"]):
        if os.path.exists(checkpoint_dir):
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
            return model, tokenizer
    
    # print("Model checkpoint not found. Downloading model from HuggingFace ...")
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
    
    # save the model and tokenizer
    # print("Saving model and tokenizer ...")
    model.save_pretrained(checkpoint_dir, save_config=True)
    tokenizer.save_pretrained(checkpoint_dir)
    
    return model, tokenizer 

def read_session_ids(cfg):
    # Read the session ids from the file - records all the session ids used so far
    try:
        if os.path.getsize(cfg['session_ids_fpath']) == 0:
            return set()  # Return an empty set if the file is empty
        with open(cfg['session_ids_fpath'], 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Session ids file not found at: {}".format(cfg['session_ids_fpath']))
    
def write_session_ids(cfg, session_ids):
    # Write the updated session ids to the file
    with open(cfg['session_ids_fpath'], 'wb') as f:
        pickle.dump(session_ids, f)
    
def generate_session_id(cfg):
    '''
    Generate a unique session id for the current chatbot session.
    Attempt to generate a unique session id 25 times before raising an error.
    Uniqueness means the session id is not stored in past_session_is.
    Args:
        cfg (dict): Configuration dictionary
    Returns:
        session_id (str): Unique session id
    Raises:
        ValueError: If a unique session id could not be generated
    '''
    past_session_ids = read_session_ids(cfg)
    
    for _ in range(25):
        session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if session_id not in past_session_ids:
            past_session_ids.add(session_id)
            write_session_ids(cfg, past_session_ids)
            
            return session_id
    
    raise ValueError("Could not generate unique session id. Please try again !!!")

def save_conversations(cfg, convos):
    # Save the conversation logs from current session
    session_conv_fpath = os.path.join(cfg['session_store_dir'], "conversation.json")
    
    with open(session_conv_fpath, 'w') as f:
        json.dump(convos, f)
    
def save_config(cfg):
    # Save the configuration dictionary with which the chatbot was initialized
    config_file = os.path.join(cfg['session_store_dir'], "config.json")
    with open(config_file, 'a') as f:
        json.dump(cfg, f)
    # print("Config saved at: {}".format(config_file))


def verify_config(cfg, model, tokenizer):
    if cfg['max_len'] > model.config.n_ctx:
        raise ValueError("'max_op_len' in cfg must be <= model.config.n_ctx")
    if cfg['max_input_prop'] + cfg['max_history_input_prop'] != 1:
        raise ValueError("'max_input_prop' + 'max_history_input_prop' must be equal to 1")
    if cfg['min_hist_input_prop'] > cfg['max_hist_input_prop']:
        raise ValueError("'min_hist_input_prop' must be <= 'max_hist_input_prop'")

def create_dirs_paths(cfg):
    '''
    Create directories and paths for storing conversation logs and models.
    Args:
        cfg (dict): Configuration dictionary
        
    Returns:
        cfg (dict): Updated configuration dictionary
    '''
    if not os.path.exists(cfg['output_store_dir']):
        os.makedirs(cfg['output_store_dir'])
    if not os.path.exists(cfg['model_store_dir']):
        os.makedirs(cfg['model_store_dir'])
    
    cfg['session_ids_fpath'] = os.path.join(cfg['output_store_dir'], "session_ids.pkl")
    if not os.path.exists(cfg['session_ids_fpath']):
        session_ids = set()
        with open(cfg['session_ids_fpath'], 'wb') as f:
            pickle.dump(session_ids, f)
    
    cfg['session_id'] = generate_session_id(cfg)
    cfg['session_store_dir'] = os.path.join(cfg['output_store_dir'], cfg['session_id'])
    if not os.path.exists(cfg['session_store_dir']):
        os.makedirs(cfg['session_store_dir'])
    
    return cfg

def create_signal_handler(cfg, cb):
    '''
    Create a signal handler for graceful when a keyboard interrupt is received - saves conversation logs and exit.
    Args:
        cfg (dict): Configuration dictionary
        cb (ChatBot): ChatBot instance
    Returns:
        signal_handler (function): Signal handler function
    '''
    def signal_handler(sig, frame):
        print("\nKeyboard interrupt received. Saving conversation logs...")
        save_conversations(cfg, cb.convos)
        print("Conversation logs saved. Exiting chatbot !!!")
        sys.exit(0)
    return signal_handler



