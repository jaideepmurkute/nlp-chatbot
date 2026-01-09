from torch.utils.data import Dataset
from system_prompts import CUSTOMER_SUPPORT_PROMPT


class ConversationDataset(Dataset):
    """
    Custom Dataset class for loading conversation data.
    Supports:
    1. 'instruction'/'response' columns (Instruction Tuning) - PREFERRED
    2. 'context'/'response' columns (Legacy/Chat)
    """

    def __init__(self, tokenizer, data, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Handle different column names
        if "instruction" in row and "response" in row:
            user_input = str(row["instruction"])
            assistant_response = str(row["response"])
        elif "context" in row and "response" in row:
            user_input = str(row["context"])
            assistant_response = str(row["response"])
        else:
            raise ValueError(
                "Dataset must have 'instruction'/'response' or 'context'/'response' columns"
            )

        system_prompt = CUSTOMER_SUPPORT_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # Create labels (Shifted logic is handled by CausalLM loss automatically,
        # so labels = input_ids is fine, but we mask padding)
        labels = input_ids.clone()

        # To ignore padding in loss calculation
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        labels[labels == pad_token_id] = -100

        # Mask the "User" instructions in the labels?
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
