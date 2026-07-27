"""
QLoRA Supervised Fine-Tuning of Qwen2.5-7B-Instruct
=====================================================

Fine-tunes Qwen2.5-7B-Instruct so it natively learns to turn GAT
attention-breakdown metrics (e.g. "Genre: 48%, Actor: 31%, Director: 17%,
Producer: 4%") into expert, natural-sounding movie recommendation
explanations — instead of relying on prompt-engineering alone
(as `llm/explainer.py` does at inference time with the base model).

This is a genuine, heavy-duty training job:
    - 4-bit NF4 QLoRA (BitsAndBytesConfig, double quantization, bf16 compute)
    - LoRA adapters on all attention + MLP projection matrices
    - trl's SFTTrainer with SFTConfig (modern API, not TrainingArguments)

Expect this to run for HOURS on a single consumer GPU (RTX 3090/4090)
or an A100 on Colab, depending on dataset size and epoch count — this
is a full parameter-efficient fine-tuning run, not a quick demo.

GPU NOTE FOR 6GB CARDS (e.g. RTX 4050 laptop GPU): this WILL work, but
is genuinely tight. The default config below (r=8 LoRA rank, attention
projections only, batch size 1, max_seq_length capped at 512) is
already tuned down from the "textbook" 24GB-card settings specifically
for this reason. If you still hit `CUDA out of memory`:
    1. Set target_modules=["q_proj", "v_proj"] only (2 modules instead of 4)
    2. Lower MAX_SEQ_LENGTH to 256 or 128
    3. Lower LoraConfig r to 4
    4. Close every other GPU-using application (browser hardware
       acceleration, other Python processes) before running
On 6GB, training will be noticeably slower per step than on a 24GB
card (gradient checkpointing recomputes activations to save memory,
at the cost of extra compute) -- multi-hour training times are
expected and are a sign this is working correctly, not stuck.

Run:
    python llm/finetune/train_qlora.py

Requires (add to requirements.txt / install separately):
    pip install -U transformers peft bitsandbytes datasets trl accelerate
"""

import os
import json
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "qwen25-7b-gat-explainer-lora")
LOSS_LOG_PATH = os.path.join(OUTPUT_DIR, "loss_history.json")
STATUS_PATH = os.path.join(OUTPUT_DIR, "training_status.json")


class LossHistoryCallback(TrainerCallback):
    """Records (step, loss, timestamp) for every logged training step, and
    writes it to disk incrementally so the Streamlit UI can display a real
    loss curve for this fine-tuning run — the same proof-of-training
    pattern used for the GAT's training diagnostics."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.history = []
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.history.append({
                "step": state.global_step,
                "epoch": logs.get("epoch"),
                "loss": logs["loss"],
                "elapsed_seconds": time.time() - self.start_time,
            })
            with open(self.log_path, "w") as f:
                json.dump(self.history, f)
MAX_SEQ_LENGTH = 1024

SYSTEM_PROMPT = (
    "You are an expert film critic and recommendation assistant. You are "
    "given attention-based graph metrics produced by a Graph Attention "
    "Network (GAT) movie recommender, showing how much each relationship "
    "type (Genre, Actor, Director, Producer) contributed to a recommendation. "
    "Write a short, natural, expert movie-critique-style explanation that "
    "references these metrics without sounding like you are just reading "
    "off numbers."
)

# ---------------------------------------------------------------------------
# 1. Mock dataset — 3 examples showing the expected format.
#    In production, replace this with `load_dataset("json", data_files=...)`
#    pointed at your real 5,000+ example file (see README section at the
#    bottom of this repo / the docstring in build_dataset_template.py).
# ---------------------------------------------------------------------------
RAW_EXAMPLES = [
    {
        "source_movie": "Interstellar",
        "recommended_movie": "Arrival",
        "metrics": {"genre": 0.49, "director": 0.28, "actor": 0.19, "producer": 0.04},
        "explanation": (
            "If Interstellar pulled you in, Arrival is a natural next stop — "
            "both lean hard into cerebral, emotionally grounded science "
            "fiction, which is the single biggest reason the model surfaced "
            "it (genre similarity: 49%). Denis Villeneuve's directing style "
            "shares real DNA with Nolan's — a patient, atmospheric approach "
            "to big ideas — contributing another 28% to the match. There's a "
            "smaller but real cast overlap pulling in the remaining "
            "similarity, while the producer connection barely moves the "
            "needle here."
        ),
    },
    {
        "source_movie": "The Dark Knight",
        "recommended_movie": "Se7en",
        "metrics": {"genre": 0.31, "director": 0.09, "actor": 0.12, "producer": 0.48},
        "explanation": (
            "This one's driven almost entirely by the producer connection "
            "(48%) — the same production hand shaped both films' dark, "
            "morally complicated tone. Genre overlap (31%) reflects the "
            "shared crime-thriller backbone, while the cast (12%) and "
            "directorial (9%) links are present but secondary here. If "
            "you're drawn to The Dark Knight's grim atmosphere more than "
            "its superhero trappings, Se7en channels that same bleakness."
        ),
    },
    {
        "source_movie": "La La Land",
        "recommended_movie": "Whiplash",
        "metrics": {"genre": 0.22, "director": 0.61, "actor": 0.10, "producer": 0.07},
        "explanation": (
            "The director signal dominates this recommendation (61%) — "
            "Damien Chazelle's fingerprints are all over both films' "
            "obsessive focus on the cost of artistic ambition, even though "
            "one is a musical romance and the other a drama-thriller. "
            "Genre overlap (22%) is modest since the surface-level tones "
            "differ quite a bit, and cast/producer ties are minor "
            "contributors — this is a director-driven match through and "
            "through."
        ),
    },
]


def format_example(example):
    """Turns one raw example into a Qwen chat-template-formatted training string."""
    metrics = example["metrics"]
    metrics_lines = "\n".join(
        f"- {k.capitalize()}: {v * 100:.0f}%" for k, v in
        sorted(metrics.items(), key=lambda kv: -kv[1])
    )
    user_content = (
        f"Source movie: {example['source_movie']}\n"
        f"Recommended movie: {example['recommended_movie']}\n"
        f"GAT attention breakdown:\n{metrics_lines}\n\n"
        f"Write the explanation."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["explanation"]},
    ]


def build_dataset(tokenizer):
    texts = []
    for ex in RAW_EXAMPLES:
        messages = format_example(ex)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def main():
    # -----------------------------------------------------------------
    # 2. Tokenizer
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------
    # 3. 4-bit NF4 quantized base model (QLoRA)
    # -----------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required for gradient checkpointing during training
    model = prepare_model_for_kbit_training(model)

    # -----------------------------------------------------------------
    # 4. LoRA adapter configuration
    #
    # NOTE ON GPU SIZE: on a 6GB card (e.g. RTX 4050 laptop GPU), the
    # 4-bit base model alone uses ~4.5GB, leaving only ~1.5GB for
    # activations, LoRA adapter weights, optimizer states, and the KV
    # cache during training. r=16 across all 7 projection types (the
    # "textbook" config, fine on a 24GB 3090/4090) can OOM here. We
    # default to a leaner config: lower rank, and attention projections
    # only (skip the MLP gate/up/down projections, which are the
    # largest matrices in the model). If you still OOM, first try
    # target_modules=["q_proj", "v_proj"] only, then reduce
    # MAX_SEQ_LENGTH further (128), then reduce r to 4.
    # -----------------------------------------------------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # -----------------------------------------------------------------
    # 5. Dataset (swap RAW_EXAMPLES / build_dataset for your real data)
    # -----------------------------------------------------------------
    train_dataset = build_dataset(tokenizer)
    print(f"Training examples: {len(train_dataset)}")
    print("Example formatted text:\n", train_dataset[0]["text"][:500], "...\n")

    # -----------------------------------------------------------------
    # 6. Training configuration (modern trl SFTConfig, not TrainingArguments)
    #
    # Batch size 1 + gradient accumulation (not batch size 2+) and a
    # shorter sequence length are the two biggest VRAM levers on a 6GB
    # card. bf16=True is kept since the RTX 4050 (Ada Lovelace) supports
    # bf16 natively; if you hit numerical issues, switch to fp16=True
    # instead. gradient_checkpointing trades compute time for a large
    # activation-memory reduction -- essential here, not optional.
    # -----------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,      # effective batch size = 16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_seq_length=min(MAX_SEQ_LENGTH, 512),  # 512 is safer than 1024 on 6GB
        dataset_text_field="text",
        packing=False,
    )

    # -----------------------------------------------------------------
    # 7. Trainer
    # -----------------------------------------------------------------
    loss_callback = LossHistoryCallback(LOSS_LOG_PATH)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        callbacks=[loss_callback],
    )

    print("Starting QLoRA fine-tuning — this is a real training job and "
          "will take a long time on a full-scale dataset (hours on a "
          "single consumer GPU). With only the 3 mock examples above it "
          "will finish almost immediately; scale up the dataset to see "
          "realistic training time.")
    trainer.train()

    # -----------------------------------------------------------------
    # 8. Save LoRA adapter (NOT the full merged model — small, portable)
    # -----------------------------------------------------------------
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")

    elapsed_total = time.time() - loss_callback.start_time
    status = {
        "adapter_dir": OUTPUT_DIR,
        "num_examples": len(train_dataset),
        "num_epochs": training_args.num_train_epochs,
        "elapsed_seconds": elapsed_total,
        "final_loss": loss_callback.history[-1]["loss"] if loss_callback.history else None,
        "first_loss": loss_callback.history[0]["loss"] if loss_callback.history else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Training status written to: {STATUS_PATH}")


if __name__ == "__main__":
    main()
