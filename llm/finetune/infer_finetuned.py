"""
Loads the base Qwen2.5-7B-Instruct model + your trained LoRA adapter
for inference, and generates an explanation from GAT attention metrics
— the fine-tuned counterpart to `llm/explainer.py`'s prompt-only version.

Two loading modes are shown:
    1. Quantized + adapter on top (lowest VRAM, matches training setup)
    2. Merged full-precision model (slower to prepare, faster to run
       repeatedly, no PEFT dependency needed at serve time)

Run:
    python llm/finetune/infer_finetuned.py
"""

import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "qwen25-7b-gat-explainer-lora")

SYSTEM_PROMPT = (
    "You are an expert film critic and recommendation assistant. You are "
    "given attention-based graph metrics produced by a Graph Attention "
    "Network (GAT) movie recommender, showing how much each relationship "
    "type (Genre, Actor, Director, Producer) contributed to a recommendation. "
    "Write a short, natural, expert movie-critique-style explanation that "
    "references these metrics without sounding like you are just reading "
    "off numbers."
)


def load_quantized_with_adapter():
    """Mode 1: 4-bit base model + LoRA adapter layered on top (matches training)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    return model, tokenizer


def load_merged_fp16():
    """Mode 2: merge LoRA weights into the base model once, save as a
    standalone fp16 model — no PEFT/bitsandbytes needed to serve it
    afterward, at the cost of full fp16 VRAM usage (~15GB) and a one-time
    merge step.

    WARNING: ~15GB of VRAM is required for this mode. On a 6GB card
    (e.g. RTX 4050) this will NOT fit on the GPU -- it will either OOM
    or silently spill onto CPU RAM via device_map="auto" and run very
    slowly. Use load_quantized_with_adapter() instead for local
    inference on this hardware; this merged mode is realistic only on a
    GPU with >=16GB VRAM or when you specifically want a portable
    standalone model to deploy elsewhere."""
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    merged = model.merge_and_unload()

    merged_dir = os.path.join(os.path.dirname(__file__), "qwen25-7b-gat-explainer-merged")
    merged.save_pretrained(merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged fp16 model saved to: {merged_dir}")
    return merged, tokenizer


def build_prompt(source_title, recommended_title, metrics):
    lines = "\n".join(
        f"- {k.capitalize()}: {v * 100:.0f}%" for k, v in
        sorted(metrics.items(), key=lambda kv: -kv[1])
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Source movie: {source_title}\n"
            f"Recommended movie: {recommended_title}\n"
            f"GAT attention breakdown:\n{lines}\n\n"
            f"Write the explanation."
        )},
    ]


def generate_explanation(model, tokenizer, source_title, recommended_title, metrics, max_new_tokens=200):
    messages = build_prompt(source_title, recommended_title, metrics)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True,
        )
    output_ids = generated[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


if __name__ == "__main__":
    if not os.path.exists(ADAPTER_DIR):
        raise SystemExit(
            f"No adapter found at {ADAPTER_DIR}. Run train_qlora.py first."
        )

    model, tokenizer = load_quantized_with_adapter()

    example_metrics = {"genre": 0.48, "actor": 0.31, "director": 0.17, "producer": 0.04}
    explanation = generate_explanation(model, tokenizer, "Interstellar", "Arrival", example_metrics)
    print(explanation)
