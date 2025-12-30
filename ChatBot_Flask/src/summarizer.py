
from typing import List, Dict
import torch

class Summarizer:
    def __init__(self, model, tokenizer, device, max_summary_length=200):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_summary_length = max_summary_length
        
    def summarize(self, current_summary: str, old_messages: List[Dict]) -> str:
        """
        Generates a new summary by combining the current summary with a batch of old messages.
        """
        # Format the old messages into a string
        conversation_text = ""
        for msg in old_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
            
        # Construct the Prompt
        # We use a carefully engineered prompt to ensure factual density.
        if current_summary:
            prompt = (
                f"You are a helpful assistant. Update the following summary of a conversation by adding the new messages.\n\n"
                f"Current Summary:\n{current_summary}\n\n"
                f"New Messages:\n{conversation_text}\n\n"
                f"Update the summary to include the new information. Keep it concise (under {self.max_summary_length} words). "
                f"Preserve key details like names, numbers, and specific instructions.\n"
                f"Updated Summary:"
            )
        else:
            prompt = (
                f"You are a helpful assistant. Summarize the following conversation.\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                f"Create a concise summary (under {self.max_summary_length} words). "
                f"Preserve key details like names, numbers, and specific instructions.\n"
                f"Summary:"
            )
            
        # Generate
        # We use the same model instance. 
        # Note: We assume the model is an Instruct model (like Qwen).
        
        # Apply Chat Template? 
        # Actually, for summarization, we can just use the raw string or a simple user message structure 
        # depending on if the model rigidly enforces chat templates. 
        # Qwen instructs usually work best within the template.
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_summary_length + 50, # Buffer for tokens
                do_sample=True,
                temperature=0.3, # Low temp for factual consistency
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_len:]
        new_summary = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return new_summary
