from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
_device = "cpu"
_torch = torch 
_dtype = _torch.float32

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    dtype=_dtype,
    device_map="auto"
)
model.eval()

def _encode(messages: list[dict]):
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(_device)
    except Exception as e:
        print(f"Error occurred while encoding messages: {e}")
        raise
    
def generate_response():
    messages = [
        {"role": "user", "content": "hi "}
    ]
    
    inputs = _encode(messages)
    print(f"Inputs: {inputs}")
    print(f"Input IDs: {inputs['input_ids']}")
    print(inputs['input_ids'].shape)
    with _torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        
if __name__ == "__main__":
    generate_response()