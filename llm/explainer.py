"""
Turns the GAT's learned attention breakdown into a natural-language
explanation of why a movie was recommended. Three backends are
supported, in increasing order of setup cost:

    "rule_based" -- fast, no LLM at all, always available.
    "ollama"     -- calls a locally running Ollama server
                    (http://localhost:11434) using whatever model you
                    already have pulled there (e.g. `ollama pull
                    qwen2.5:7b` / `llama3.1` / etc.). This is the
                    recommended local-LLM option: no HuggingFace
                    download, no manual quantization config, and it
                    reuses a model you likely already have running.
    "qwen"       -- loads Qwen2.5-7B-Instruct locally via HuggingFace
                    transformers (no external APIs). This is a 7B
                    model needing real GPU VRAM, kept as a heavier
                    alternative for machines set up for it.

Both "ollama" and "qwen" are loaded/contacted lazily (only when the
user actually asks for an explanation), and both fall back to the
rule-based explanation on any error so the rest of the app stays
usable regardless of what's locally available.
"""

import os
import json
import threading
import urllib.request
import urllib.error

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "finetune", "qwen25-7b-gat-explainer-lora")
STATUS_PATH = os.path.join(ADAPTER_DIR, "training_status.json")
LOSS_LOG_PATH = os.path.join(ADAPTER_DIR, "loss_history.json")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

_model = None
_tokenizer = None
_loaded_variant = None  # "base" or "finetuned" — tracks which one is currently in memory
_lock = threading.Lock()


def finetuned_adapter_available():
    """True once train_qlora.py has actually completed and saved an adapter."""
    return os.path.exists(ADAPTER_DIR) and os.path.exists(STATUS_PATH)


def load_finetune_status():
    """Returns the training_status.json dict written by train_qlora.py, or None."""
    if not os.path.exists(STATUS_PATH):
        return None
    with open(STATUS_PATH) as f:
        return json.load(f)


def load_finetune_loss_history():
    """Returns the list of {step, epoch, loss, elapsed_seconds} dicts recorded
    during fine-tuning, or None if no run has completed yet."""
    if not os.path.exists(LOSS_LOG_PATH):
        return None
    with open(LOSS_LOG_PATH) as f:
        return json.load(f)


def _load_model(use_finetuned=False):
    global _model, _tokenizer, _loaded_variant
    target_variant = "finetuned" if use_finetuned else "base"

    with _lock:
        if _model is not None and _loaded_variant == target_variant:
            return _model, _tokenizer

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # 4-bit NF4 loading -- required on 6-8GB consumer GPUs (e.g. RTX
        # 4050/4060). Loading Qwen2.5-7B at fp16 needs ~14GB of VRAM just
        # for weights, which will not fit on cards in that class; 4-bit
        # quantization brings the weight footprint down to ~4-4.5GB,
        # leaving headroom for the KV cache during generation.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        if use_finetuned:
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_config, device_map="auto",
            )
            _model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
        else:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=bnb_config, device_map="auto",
            )
        _loaded_variant = target_variant
    return _model, _tokenizer


def ollama_available():
    """True if a local Ollama server is reachable at OLLAMA_HOST."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models():
    """Returns the list of model names currently pulled in the local
    Ollama install (e.g. ['qwen2.5:7b', 'llama3.1:8b']), or [] if the
    server isn't reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _call_ollama(messages, model=None, timeout=60):
    model = model or DEFAULT_OLLAMA_MODEL
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["message"]["content"].strip()


def is_model_loaded():
    return _model is not None


def currently_loaded_variant():
    return _loaded_variant


def _build_prompt(source_title, recommended_title, breakdown, method):
    contribution = breakdown.get("normalized_contribution", {})
    lines = [f"{k.capitalize()} contributed {v * 100:.0f}%" for k, v in
              sorted(contribution.items(), key=lambda x: -x[1])]
    contribution_text = "\n".join(lines) if lines else "No attention data available."

    prompt = f"""You are an assistant that explains movie recommendations produced by a
Graph Attention Network (GAT) recommender system.

Source movie: {source_title}
Recommended movie: {recommended_title}
Recommendation method: {method}

The GAT model learned the following attention-based contribution
scores when aggregating information from the movie's graph
neighborhood (genre, actor, director, and same-franchise/title nodes):

{contribution_text}

Write a short (3-4 sentence), friendly explanation for a student
audience describing why "{recommended_title}" was recommended given
"{source_title}", referencing the attention percentages above and
what they mean about which relationships mattered most. Do not
invent facts not present above."""
    return prompt


# Must exactly match the SYSTEM_PROMPT / user-message format used in
# llm/finetune/train_qlora.py — the fine-tuned adapter was trained on
# this specific format, not the longer base-model prompt above.
_FINETUNE_SYSTEM_PROMPT = (
    "You are an expert film critic and recommendation assistant. You are "
    "given attention-based graph metrics produced by a Graph Attention "
    "Network (GAT) movie recommender, showing how much each relationship "
    "type (Genre, Actor, Director, Same-Franchise/Title) contributed to a "
    "recommendation. Write a short, natural, expert movie-critique-style "
    "explanation that references these metrics without sounding like you "
    "are just reading off numbers."
)


def _build_finetuned_messages(source_title, recommended_title, breakdown):
    contribution = breakdown.get("normalized_contribution", {})
    lines = "\n".join(
        f"- {k.capitalize()}: {v * 100:.0f}%" for k, v in
        sorted(contribution.items(), key=lambda kv: -kv[1])
    )
    user_content = (
        f"Source movie: {source_title}\n"
        f"Recommended movie: {recommended_title}\n"
        f"GAT attention breakdown:\n{lines}\n\n"
        f"Write the explanation."
    )
    return [
        {"role": "system", "content": _FINETUNE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def explain_recommendation(source_title, recommended_title, breakdown, method="GAT",
                            backend="rule_based", ollama_model=None,
                            use_llm=None, use_finetuned=False):
    """
    backend: one of "rule_based", "ollama", "qwen".

    For backward compatibility, the old `use_llm=True/False` flag is
    still accepted: `use_llm=False` forces "rule_based"; `use_llm=True`
    (with backend left at its default) maps to "qwen" so existing
    callers keep working unchanged.
    """
    if use_llm is False:
        backend = "rule_based"
    elif use_llm is True and backend == "rule_based":
        backend = "qwen"

    if backend == "rule_based":
        return _rule_based_explanation(source_title, recommended_title, breakdown, method)

    if backend == "ollama":
        return _explain_via_ollama(source_title, recommended_title, breakdown, method, ollama_model)

    if backend == "qwen":
        return _explain_via_qwen(source_title, recommended_title, breakdown, method, use_finetuned)

    # Unknown backend -> safest default.
    return _rule_based_explanation(source_title, recommended_title, breakdown, method)


def _explain_via_ollama(source_title, recommended_title, breakdown, method, ollama_model=None):
    try:
        prompt = _build_prompt(source_title, recommended_title, breakdown, method)
        messages = [{"role": "user", "content": prompt}]
        return _call_ollama(messages, model=ollama_model)
    except Exception as e:  # pragma: no cover - fallback path
        fallback = _rule_based_explanation(source_title, recommended_title, breakdown, method)
        return (f"[Ollama unavailable at {OLLAMA_HOST} ({e}); is `ollama serve` running "
                f"and is the model pulled? Showing rule-based explanation instead]\n\n{fallback}")


def _explain_via_qwen(source_title, recommended_title, breakdown, method, use_finetuned=False):
    if use_finetuned and not finetuned_adapter_available():
        use_finetuned = False  # silently fall back rather than error if not trained yet

    try:
        import torch
        model, tokenizer = _load_model(use_finetuned=use_finetuned)

        if use_finetuned:
            messages = _build_finetuned_messages(source_title, recommended_title, breakdown)
        else:
            prompt = _build_prompt(source_title, recommended_title, breakdown, method)
            messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
            )

        output_ids = generated[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip()

    except Exception as e:  # pragma: no cover - fallback path
        fallback = _rule_based_explanation(source_title, recommended_title, breakdown, method)
        return f"[Local Qwen LLM unavailable ({e}); showing rule-based explanation]\n\n{fallback}"


def _rule_based_explanation(source_title, recommended_title, breakdown, method):
    contribution = breakdown.get("normalized_contribution", {})
    if not contribution:
        return (f"'{recommended_title}' was recommended based on '{source_title}' "
                f"using the {method} method, but no attention breakdown is available.")

    sorted_items = sorted(contribution.items(), key=lambda x: -x[1])
    top_type, top_score = sorted_items[0]

    parts = [f"{k.capitalize()} contributed {v * 100:.0f}%" for k, v in sorted_items]
    contribution_sentence = ", ".join(parts)

    return (
        f"'{recommended_title}' was recommended because of its similarity to "
        f"'{source_title}' as learned by the {method} model. The dominant factor was "
        f"{top_type} similarity ({top_score * 100:.0f}%). Full attention breakdown: "
        f"{contribution_sentence}."
    )