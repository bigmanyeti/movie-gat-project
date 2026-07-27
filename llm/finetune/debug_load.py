"""
Minimal isolation script to find exactly where model loading crashes.
Run: python debug_load.py
Watch which numbered print is the LAST one to appear before it dies.
"""
import sys

print("1. starting", flush=True)

import torch
print(f"2. torch imported, version={torch.__version__}", flush=True)
print(f"2b. cuda available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"2c. GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"2d. Free/Total VRAM (bytes): {torch.cuda.mem_get_info()}", flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print("3. transformers imported", flush=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
print("4. tokenizer loaded", flush=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
print("5. bnb_config created", flush=True)

print("6. about to call from_pretrained -- this is the likely crash point", flush=True)
sys.stdout.flush()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
print("7. MODEL LOADED SUCCESSFULLY", flush=True)

print(f"8. Free/Total VRAM after load (bytes): {torch.cuda.mem_get_info()}", flush=True)