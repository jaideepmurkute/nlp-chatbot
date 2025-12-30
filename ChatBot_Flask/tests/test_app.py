
import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------------------
# MOCKING STRATEGY
# We must mock 'chatbot' module BEFORE importing 'app' 
# to prevent it from loading the heavy ML models.
# ----------------------------------------------------------------------

# 1. Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 2. Create the Mock Class
class MockChatBot:
    def __init__(self, cfg, app):
        self.cfg = cfg
        self.app = app
        self.last_response = "Mock Response"
        self.context = MagicMock()
        self.context.session_metadata = {'session_id': 'test_session'}
        self.context.logs = []
        
        # Trigger route registration just like the real class
        self._setup_routes()
        
    def _setup_routes(self):
        # We must register the urls to the app instance
        self.app.add_url_rule('/', view_func=self._index)
        self.app.add_url_rule('/model_info', methods=['GET'], view_func=self._model_info)
        self.app.add_url_rule('/chat', methods=['POST'], view_func=self._chat)
        self.app.add_url_rule('/new_session', methods=['POST'], view_func=self._new_session)

    def _index(self):
        return "<html>Index</html>", 200

    def _model_info(self):
        from flask import jsonify
        return jsonify(model_name=self.cfg.get('model_name', 'test_model'))

    def _chat(self):
        from flask import request, jsonify
        user_input = request.form['user_input']
        self.generate_response(user_input)
        return jsonify(response=self.last_response)

    def _new_session(self):
        from flask import jsonify
        return jsonify(success=True)

    def generate_response(self, user_input):
        self.last_response = f"Echo: {user_input}"
        return

# 3. Apply the Mock to sys.modules
# When app.py runs 'from chatbot import ChatBot', it will get our MockChatBot
mock_chatbot_module = MagicMock()
mock_chatbot_module.ChatBot = MockChatBot
sys.modules['chatbot'] = mock_chatbot_module

# ----------------------------------------------------------------------
# NOW IMPORT APP
# ----------------------------------------------------------------------
# We also need to mock signal handling to avoid "ValueError: signal only works in main thread" 
# issues during some test runners, though unittest usually handles it.
# But let's be safe.
with patch('signal.signal'):
    from app import app, cb

class TestFlaskApp(unittest.TestCase):
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_index_route(self):
        """Test the home page loads successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        # Check if mocked template rendered
        self.assertIn(b'Index', response.data)

    def test_model_info_route(self):
        """Test the model info endpoint."""
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('model_name', data)

    def test_chat_interaction(self):
        """Test sending a message and getting a response."""
        user_input = "Hello Robot"
        response = self.app.post('/chat', data={'user_input': user_input})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify our MockChatBot logic was called
        # Our mock says "Echo: {input}"
        self.assertEqual(data['response'], "Echo: Hello Robot")
        
        # Verify global callback state
        self.assertEqual(cb.last_response, "Echo: Hello Robot")

    def test_new_session(self):
        """Test resetting the session."""
        response = self.app.post('/new_session')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])

if __name__ == '__main__':
    unittest.main()
