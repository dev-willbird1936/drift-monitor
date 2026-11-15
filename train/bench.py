"""3080 viability benchmark: 4-bit QLoRA training and scoring speed on real prompts. usage: bench.py MODEL TRAIN_JSONL"""
import json, sys, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
mid, path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(mid)
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
random.Random(0).shuffle(rows)
enc = lambda r: tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True) + r["output"]
lens = sorted(len(tok(enc(r)).input_ids) for r in rows[:1500])
q = lambda p: lens[int(p * (len(lens) - 1))]
print("token lengths p50/p90/p99/max:", q(.5), q(.9), q(.99), lens[-1], flush=True)
model = AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
print("loaded 4-bit, VRAM GB:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
# scoring speed: one forward pass, read next-token logits
model.eval()
for L in (q(.5), q(.99)):
    r = min(rows[:1500], key=lambda r: abs(len(tok(enc(r)).input_ids) - L))
    ids = tok(enc(r), return_tensors="pt").input_ids[:, :-1].cuda()
    with torch.no_grad():
        model(ids); torch.cuda.synchronize(); t = time.time()
        for _ in range(5): model(ids).logits[0, -1]
        torch.cuda.synchronize()
    print(f"score: {ids.shape[1]} tokens -> {(time.time() - t) / 5:.2f} s per score", flush=True)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
model.train()
for L in (q(.5), q(.9), lens[-1]):
    r = min(rows[:1500], key=lambda r: abs(len(tok(enc(r)).input_ids) - L))
    ids = tok(enc(r), return_tensors="pt").input_ids.cuda()
    labels = torch.full_like(ids, -100); labels[0, -1] = ids[0, -1]
    torch.cuda.reset_peak_memory_stats(); ts = []
    try:
        for i in range(5):
            t = time.time(); loss = model(input_ids=ids, labels=labels).loss; loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize(); ts.append(time.time() - t)
        print(f"train: {ids.shape[1]} tokens -> {sum(ts[1:]) / 4:.2f} s per example, peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"train: {ids.shape[1]} tokens -> OUT OF MEMORY", flush=True); torch.cuda.empty_cache()
