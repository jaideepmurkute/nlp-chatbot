
from model_singleton import ModelSingleton
from CFG import Config

cfg = Config().config
ms = ModelSingleton(cfg)
tokenizer = ms.tokenizer
print(f"EOS Token: {tokenizer.eos_token}")
print(f"EOS Token ID: {tokenizer.eos_token_id}")
print(f"Token 50256: {tokenizer.decode([50256])}")
print(f"Is 50256 special? {50256 in tokenizer.all_special_ids}")
