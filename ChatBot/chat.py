from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from CFG import Config

def get_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token
    
    return model, tokenizer 

def chat(cfg, model, tokenizer):
    bot_input_ids = torch.tensor([])
    bot_attention_mask = torch.tensor([])
    
    for step in range(5):
        user_input = input(">> User:")
        user_input_encoded = tokenizer.encode_plus(
                        user_input + tokenizer.eos_token,
                        return_tensors='pt',
                        padding=True,
                        truncation=True
                    )
        
        if step == 0:
            bot_input_ids = user_input_encoded['input_ids']
            bot_attention_mask = user_input_encoded['attention_mask']
        else:
            bot_input_ids = torch.cat([bot_input_ids, \
                                    user_input_encoded['input_ids']], dim=-1) 
            bot_attention_mask = torch.cat([bot_attention_mask, \
                                    user_input_encoded['attention_mask']], dim=-1)
        
        model_op_ids = model.generate(bot_input_ids, 
                                attention_mask=bot_attention_mask, 
                                max_length=cfg['max_response_len'], \
                                pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(model_op_ids[:, bot_input_ids.shape[-1]:][0], \
                                    skip_special_tokens=True)
        
        print("Model: {}".format(response))
        
        # Encode the model's response to get the attention mask
        response_encoded = tokenizer.encode_plus(
            response + tokenizer.eos_token,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        bot_input_ids = torch.cat([bot_input_ids, response_encoded['input_ids']], 
                                  dim=-1)
        bot_attention_mask = torch.cat([bot_attention_mask, \
                                response_encoded['attention_mask']], dim=-1)
        

if __name__ == "__main__":
    config = Config()
    cfg = config.config
    
    model, tokenizer = get_model_and_tokenizer(cfg)
    chat = chat(cfg, model, tokenizer)
    chat.infer()
