
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
from tqdm import tqdm
from config import Config


def load_model(cfg):
    base_save_path = os.path.join(cfg['model_store_dir'], cfg['experiment_name'], 'finetuned_model')
    model_save_path = os.path.join(base_save_path, cfg['checkpoint_type'])
    print(f"Loading {cfg['checkpoint_type']} model from {model_save_path}...")
    
    if not os.path.exists(model_save_path):
        print(f"Model load path does not exist: {model_save_path}")
        # Logic to fallback could go here, but for strict testing we might want to fail or warn
        # For now, let's assume valid config or let transformers error out if path is bad
        # Falling back to base path if specific checkpoint invalid:
        if os.path.exists(base_save_path):
             print(f"Falling back to base path: {base_save_path}")
             model_save_path = base_save_path
        else:
             print(f"Falling back to base model name: {cfg['model_name']}")
             model_save_path = cfg['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    
    device = torch.device(cfg['device'])
    model.to(device)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, context, device, max_len=1000):
     # Encode context
    new_user_input_ids = tokenizer.encode(context + tokenizer.eos_token, return_tensors='pt').to(device)

    # Generate response
    # append the new user input tokens to the chat history
    # set top_k=50 and top_p=0.95 for top-p (nucleus) sampling
    # set temperature=0.7 for some randomness
    chat_history_ids = model.generate(
        new_user_input_ids, 
        max_length=max_len,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=50, 
        top_p=0.95,
        temperature=0.7
    )

    # Decode response
    # skip the input tokens to get only the generated response
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response


def test(cfg):
    device = torch.device(cfg['device'])
    model, tokenizer = load_model(cfg)
    
    print(f"Loading test data from {cfg['data_path']}...")
    try:
        df = pd.read_csv(cfg['data_path'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {cfg['data_path']}")
        return

    print("Generating responses for test set...")
    results = []
    
    # Iterate through the dataset
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        context = str(row['context'])
        true_response = str(row['response']) if 'response' in row else ""
        
        # Generate prediction
        generated_response = generate_response(model, tokenizer, context, device, cfg['max_len'])
        
        results.append({
            'context': context,
            'true_response': true_response,
            'generated_response': generated_response
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Use model_store_dir/experiment_name for output
    output_dir = os.path.join(cfg['model_store_dir'], cfg['experiment_name'])
    
    output_filename = f"test_predictions_{cfg['checkpoint_type']}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nTest completed. Results saved to {output_path}")


if __name__ == "__main__":
    cfg = Config().get_config()
    test(cfg)
