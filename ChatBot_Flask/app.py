"""
NLP Chatbot Application

Author: Jaideep Murkute
Date: XXXX-XX-XX
Version: 1.0

"""

from flask import Flask

from CFG import Config
from chatbot import ChatBot
from utils import create_dirs_paths, save_config, save_conversations


# ------------------ Main Application ------------------ #

app = Flask(__name__)

cfg = Config().get_config()
cfg = create_dirs_paths(cfg)
save_config(cfg)

cb = ChatBot(cfg, app)

if __name__ == "__main__":
    try:
        # app.run(debug=True)
        app.run(host="0.0.0.0", port=5100)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving conversation logs...")
        save_conversations(cfg, cb.convos)
        print("Conversation logs saved. Exiting chatbot !!!")

    