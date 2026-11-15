"""Build the balanced fine-tuning set from the CuratorKIT output. usage: prep.py DATASET_DIR OUT_DIR [--zero-share 0.38] [--max-tokens 3000]"""
import json, os, random, sys, collections
from transformers import AutoTokenizer
d, out = sys.argv[1], sys.argv[2]
arg = lambda k, v: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else v
zero_share, max_tok = arg("--zero-share", 0.38), int(arg("--max-tokens", 3000))
tok = AutoTokenizer.from_pretrained(os.environ.get("DRIFT_TOKENIZER", "./tokenizer"))
rep = {}
for split, cap in (("train", None), ("val", 400)):
    rows = [json.loads(l) for l in open(os.path.join(d, "curated", f"{split}_final.jsonl"), encoding="utf-8")]
    n0 = len(rows)
    keep = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["output"]}], tokenize=False)
        n = len(tok(text, add_special_tokens=False).input_ids)
        if n <= max_tok:
            keep.append(r)
    rng = random.Random(13)
    zeros = [r for r in keep if r["output"] == "0"]; rest = [r for r in keep if r["output"] != "0"]
    nz = min(len(zeros), int(len(rest) * zero_share / (1 - zero_share)))
    sel = rest + rng.sample(zeros, nz)
    rng.shuffle(sel)
    if cap:
        sel = sel[:cap]
    os.makedirs(os.path.join(out, split), exist_ok=True)
    with open(os.path.join(out, split, "sft.jsonl"), "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps({"messages": [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["output"]}]}, ensure_ascii=False) + "\n")
    rep[split] = {"curated": n0, "within_token_cap": len(keep), "selected": len(sel), "digits": dict(sorted(collections.Counter(r["output"] for r in sel).items()))}
json.dump(rep, open(os.path.join(out, "prep_report.json"), "w"), indent=1); print(json.dumps(rep))
