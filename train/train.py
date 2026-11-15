"""Fine-tune Tiny Aya as a drift scorer with AlignTune (TRL backend, 4-bit QLoRA).
usage: train.py DATA_DIR OUT_DIR [--max-steps N] [--epochs N] [--max-samples N]
"""
import os
import sys, json, time
from aligntune.core.backend_factory import create_sft_trainer
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
data, out = sys.argv[1], sys.argv[2]
MODEL = arg("--model", "CohereLabs/tiny-aya-global")
TOKENIZER = arg("--tokenizer", os.environ.get("DRIFT_TOKENIZER", "./tokenizer"))
FOURBIT = "--full-precision" not in sys.argv  # 4-bit base for small GPUs; full bf16 base when memory allows
# Tiny Aya ships its chat template as a dict of named templates, which TRL's assistant-only masking cannot read.
# This is the same turn format as a single string, without the long default preamble, with generation markers.
TEMPLATE = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "drift_template.jinja"), encoding="utf-8").read()
kw = dict(
    model_name=MODEL, dataset_name=data, backend="trl", output_dir=out,
    num_epochs=int(arg("--epochs", "2")), batch_size=int(arg("--batch-size", "1")), learning_rate=float(arg("--lr", "1e-4")),
    max_seq_length=int(arg("--max-seq", "2048")),
    use_lora=True, lora_r=16, lora_alpha=32, lora_dropout=0.05,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing=True, gradient_accumulation_steps=int(arg("--grad-accum", "16")), device_map={"": 0},
    curator_schema_gate=False, seed=42, assistant_only_loss=True, tokenizer_name_or_path=TOKENIZER,
    eval_strategy="no", save_strategy="steps", save_steps=25, save_interval=25, logging_steps=5, save_total_limit=4,
)
if FOURBIT:
    kw["quantization"] = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True}
else:
    kw["precision"] = "bf16"
if arg("--max-steps"):
    kw["max_steps"] = int(arg("--max-steps"))
if arg("--max-samples"):
    kw["max_samples"] = int(arg("--max-samples"))
t = time.time()
trainer = create_sft_trainer(**kw)
res = trainer.train()
print("TRAIN_DONE seconds", round(time.time() - t), json.dumps({k: v for k, v in (res or {}).items() if isinstance(v, (int, float, str))})[:600], flush=True)
