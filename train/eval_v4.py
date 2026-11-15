"""Score a v4 test split with the tuned (or base) model and report metrics.

usage: eval_v4.py DATASET_V4_DIR SPLIT OUT_JSON [--adapter DIR|none] [--base MODEL] [--tokenizer DIR]
                  [--limit N] [--threshold 40]

Reports overall error and discrimination, then breaks both down by evidence tier, by drift mode and by work
band. The per-tier breakdown is the point of the v4 format: with richer evidence the score should track the
label more closely and the output distribution should be tighter.
"""
import collections, json, math, os, sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l]
    neg = [s for l, s in zip(labels, scores) if not l]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rpos = sum(r for r, l in zip(ranks, labels) if l)
    n1, n2 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2) / (n1 * n2)


def pearson(a, b):
    if len(a) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def block(rows, thr):
    if not rows:
        return None
    pred = [r["pred"] for r in rows]
    true = [r["drift_score"] for r in rows]
    lab = [r["drift_score"] >= thr for r in rows]
    tp = sum(1 for r in rows if r["pred"] >= thr and r["drift_score"] >= thr)
    fp = sum(1 for r in rows if r["pred"] >= thr and r["drift_score"] < thr)
    fn = sum(1 for r in rows if r["pred"] < thr and r["drift_score"] >= thr)
    tn = sum(1 for r in rows if r["pred"] < thr and r["drift_score"] < thr)
    out = dict(
        n=len(rows),
        mae=round(sum(abs(p - t) for p, t in zip(pred, true)) / len(rows), 2),
        bias=round(sum(p - t for p, t in zip(pred, true)) / len(rows), 2),
        pearson=(round(pearson(pred, true), 3) if pearson(pred, true) is not None else None),
        auroc=(round(auroc(lab, pred), 3) if auroc(lab, pred) is not None else None),
        mean_spread=round(sum(r["spread"] for r in rows) / len(rows), 2),
        recall=(round(tp / (tp + fn), 3) if tp + fn else None),
        precision=(round(tp / (tp + fp), 3) if tp + fp else None),
        false_alarm=(round(fp / (fp + tn), 3) if fp + tn else None))
    return out


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    ds, split, out = sys.argv[1], sys.argv[2], sys.argv[3]
    adapter = arg("--adapter", "none")
    base = arg("--base", "CohereLabs/tiny-aya-global")
    thr = float(arg("--threshold", "40"))
    limit = int(arg("--limit", "0"))

    rows = [json.loads(l) for l in open(os.path.join(ds, "data", split + ".jsonl"), encoding="utf-8")]
    if limit and len(rows) > limit:
        import random
        rows = random.Random(7).sample(rows, limit)

    tok = AutoTokenizer.from_pretrained(arg("--tokenizer", adapter if adapter != "none" else base))
    model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    if adapter != "none":
        model = PeftModel.from_pretrained(model, adapter)
    model = model.eval()
    digits = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    centres = np.arange(10) * 10 + 5

    for i, r in enumerate(rows):
        t = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            p = torch.softmax(model(ids, use_cache=False).logits[0, -1, digits].float(), -1).cpu().numpy()
        mean = float((p * centres).sum())
        r["pred"] = mean
        r["spread"] = float(math.sqrt((p * (centres - mean) ** 2).sum()))
        torch.cuda.empty_cache()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    res = dict(split=split, adapter=adapter, threshold=thr, overall=block(rows, thr))
    res["by_tier"] = {str(t): block([r for r in rows if r["tier"] == t], thr)
                      for t in sorted({r["tier"] for r in rows})}
    res["by_band"] = {b: block([r for r in rows if r.get("band") == b], thr)
                      for b in sorted({str(r.get("band")) for r in rows})}
    modes = collections.Counter(r.get("mode") for r in rows)
    res["by_mode"] = {m: block([r for r in rows if r.get("mode") == m], thr)
                      for m, n in modes.most_common() if n >= 15}
    res["by_difficulty"] = {d: block([r for r in rows if r.get("difficulty") == d], thr)
                            for d in sorted({str(r.get("difficulty")) for r in rows})}
    json.dump(dict(summary=res, preds=[{k: r[k] for k in ("id", "tier", "mode", "band", "drift_score", "pred", "spread")}
                                       for r in rows]), open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
