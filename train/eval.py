"""Score test examples with a model and report drift-score metrics.
usage: eval.py MODEL DATA_JSONL OUT_JSON [--adapter DIR] [--limit N] [--max-tokens N]
The model answers with one digit 0-9; the score is the probability-weighted mean of (10*d + 5).
"""
import json, sys, time, random, torch, numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
mid, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
limit, max_tok = int(arg("--limit", "0")), int(arg("--max-tokens", "7800"))
tok = AutoTokenizer.from_pretrained(arg("--tokenizer", mid))
if "--full-precision" in sys.argv:
    model = AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0}, torch_dtype=torch.bfloat16)
else:
    model = AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
if arg("--adapter"):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, arg("--adapter"))
model.eval()
digit_ids = []
for d in range(10):
    ids = tok.encode(str(d), add_special_tokens=False)
    assert len(ids) == 1, f"digit {d} is not one token: {ids}"
    digit_ids.append(ids[0])
rows = [w for w in (json.loads(l) for l in open(path, encoding="utf-8")) if w["clean"]]
random.Random(0).shuffle(rows)
if limit:
    rows = rows[:limit]
pred, t0 = [], time.time()
for i, w in enumerate(rows):
    text = tok.apply_chat_template([{"role": "user", "content": w["prompt"]}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -max_tok:].cuda()
    with torch.no_grad():
        p = torch.softmax(model(ids, use_cache=False).logits[0, -1, digit_ids].float(), -1).cpu().numpy()
    pred.append(float((p * (np.arange(10) * 10 + 5)).sum()))
    if i % 500 == 0:
        print(i, len(rows), round(time.time() - t0), "s", ids.shape[1], "tok, reserved GB", round(torch.cuda.memory_reserved() / 1e9, 1), flush=True)
    del ids
pred = np.array(pred); y = np.array([w["drift_score"] for w in rows]); adv = np.array([w["advise"] for w in rows]); lab = np.array([w["label"] == "drift" for w in rows])
fire = pred >= 40
res = {"precision": "bf16" if "--full-precision" in sys.argv else "4bit", "model": mid, "adapter": arg("--adapter"), "data": path, "n": len(rows), "seconds_per_score": round((time.time() - t0) / len(rows), 3),
       "mae": round(float(np.abs(pred - y).mean()), 2), "mae_if_always_5": round(float(np.abs(5 - y).mean()), 2),
       "pearson": round(float(np.corrcoef(pred, y)[0, 1]), 3), "auroc_drift_label": round(float(roc_auc_score(lab, pred)), 3),
       "trigger40_precision": round(float(adv[fire].mean()), 3) if fire.any() else None, "trigger40_recall": round(float(fire[adv].mean()), 3),
       "mean_pred_on_course": round(float(pred[~lab].mean()), 1), "mean_pred_drift": round(float(pred[lab].mean()), 1)}
for t in ("actions", "reasoning"):
    k = np.array([w["tier"] == t for w in rows])
    if k.sum() > 20 and lab[k].any() and (~lab[k]).any():
        res[f"mae_{t}"] = round(float(np.abs(pred[k] - y[k]).mean()), 2); res[f"auroc_{t}"] = round(float(roc_auc_score(lab[k], pred[k])), 3)
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
with open(out.replace(".json", "_preds.jsonl"), "w", encoding="utf-8") as f:
    for w, p in zip(rows, pred):
        f.write(json.dumps({"id": w["id"], "pred": round(float(p), 2), "drift_score": w["drift_score"], "label": w["label"], "advise": w["advise"],
                            "category": w["category"], "condition": w["condition"], "tier": w["tier"], "task_category": w["task_category"], "agent_model": w["agent_model"]}) + "\n")
