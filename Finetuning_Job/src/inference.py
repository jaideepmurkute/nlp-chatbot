import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
from tqdm import tqdm
from config import Config
from system_prompts import CUSTOMER_SUPPORT_PROMPT


def load_model(cfg):
    base_save_path = os.path.join(
        cfg["model_store_dir"], cfg["experiment_name"], "finetuned_model"
    )
    model_save_path = os.path.join(base_save_path, cfg["checkpoint_type"])
    print(f"Loading {cfg['checkpoint_type']} model from {model_save_path}...")

    if not os.path.exists(model_save_path):
        print(f"Model load path does not exist: {model_save_path}")
        if os.path.exists(base_save_path):
            print(f"Falling back to base path: {base_save_path}")
            model_save_path = base_save_path
        else:
            print(f"Falling back to base model name: {cfg['model_name']}")
            model_save_path = cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    # --------------------------------------
    # QLoRA
    use_4bit = cfg.get("use_4bit", False)
    quantization_config = None

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            print("QLoRA Enabled: Using 4-bit quantization for Inference.")

            compute_dtype = getattr(torch, cfg.get("bnb_4bit_compute_dtype", "float16"))

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        except ImportError:
            print("Error: 'bitsandbytes' not found. Cannot Use 4-bit quantization.")
            raise

    model = AutoModelForCausalLM.from_pretrained(
        model_save_path,
        quantization_config=quantization_config,
        device_map="auto" if use_4bit else None,
    )

    device = torch.device(cfg["device"])
    if not use_4bit:
        model.to(device)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, context, device, max_len=1000):
    # Encode context
    new_user_input_ids = tokenizer.encode(
        context + tokenizer.eos_token, return_tensors="pt"
    ).to(device)

    # Generate response
    chat_history_ids = model.generate(
        new_user_input_ids,
        max_length=max_len,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Decode response
    # skip the input tokens to get only the generated response
    response = tokenizer.decode(
        chat_history_ids[:, new_user_input_ids.shape[-1] :][0], skip_special_tokens=True
    )

    return response


def test(cfg):
    device = torch.device(cfg["device"])
    model, tokenizer = load_model(cfg)

    print(f"Loading test data from {cfg['data_path']}...")
    try:
        df = pd.read_csv(cfg["data_path"])
        df = df[:10]
    except FileNotFoundError:
        print(f"Error: Data file not found at {cfg['data_path']}")
        return

    print("Generating responses for test set...")
    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
        # columns names are different in two datasets
        if "instruction" in row:
            user_input = str(row["instruction"])
            true_response = str(row["response"])
        elif "context" in row:
            user_input = str(row["context"])
            true_response = str(row["response"])
        else:
            continue

        # -----------------------------------
        messages = [
            {"role": "system", "content": CUSTOMER_SUPPORT_PROMPT},
            {"role": "user", "content": user_input},
        ]

        # "add_generation_prompt=True": Qwen will append "<|im_start|>assistant\n" at the end
        # so it knows it is its turn to speak.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Generate
        """
        Setting `pad_token_id=tokenizer.eos_token_id`: since some models may not have a padding token or 
            some sequences may end early in batch inference mode and not reach the max_new_tokens limit.
        """
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens (the response)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results.append(
            {
                "input": user_input,
                "true_response": true_response,
                "generated_response": model_response,
            }
        )

    results_df = pd.DataFrame(results)

    output_dir = os.path.join(cfg["model_store_dir"], cfg["experiment_name"])
    output_filename = f"test_predictions_{cfg['checkpoint_type']}.csv"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    print("\nResults df saved to:", output_path)

    print("Testing completed.")


if __name__ == "__main__":
    cfg = Config().get_config()
    test(cfg)
