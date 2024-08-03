"""
NLP Chatbot Application

Author: Jaideep Murkute
Date: XXXX-XX-XX
Version: 1.0

"""

import signal
import sys
from flask import Flask

from CFG import Config
from chatbot import ChatBot
from utils import create_dirs_paths, save_config, save_conversations, create_signal_handler


# ------------------ Main Application ------------------ #

# Initialize the Flask application
app = Flask(__name__)

# Load the configuration settings
cfg = Config().get_config()
cfg = create_dirs_paths(cfg)
save_config(cfg)

# Initialize the ChatBot
cb = ChatBot(cfg, app)

# To handle keyboard interrupts gracefully
signal_handler = create_signal_handler(cfg, cb)
signal.signal(signal.SIGINT, signal_handler) 


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5100) # debug=True
    except Exception as e:
        print(f"An error occurred: {e}")
    
    