from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = 'unsloth/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="mps")
messages = [{'role': 'user', 'content': 'hello'}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True)
input_ids = inputs["input_ids"].to("mps")
attention_mask = inputs["attention_mask"].to("mps")

model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)
