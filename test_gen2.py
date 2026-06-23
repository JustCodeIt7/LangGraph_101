from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = 'unsloth/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
messages = [{'role': 'user', 'content': 'hello'}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
print(type(inputs))
