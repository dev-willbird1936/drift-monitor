"""CuratorKIT pass over the exact training split the final model was trained on (run after training, as an audit).

usage: audit_v4b.py TRAIN_DIR OUT_DIR      (TRAIN_DIR holds train/sft.jsonl and val/sft.jsonl from prep_v4.py)
Reports what cleaning, MinHash dedup and the secrets gate would keep or reject. Thresholds match the v3 pass.
"""
import collections, json, os, sys
from curatorkit import Curator, CuratorConfig

src_dir, out = sys.argv[1], sys.argv[2]
rep = {}
for split in ("train", "val"):
    rows = [json.loads(l) for l in open(os.path.join(src_dir, split, "sft.jsonl"), encoding="utf-8")]
    os.makedirs(out, exist_ok=True)
    src = os.path.join(out, f"{split}_input.jsonl")
    with open(src, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            m = r["messages"]
            f.write(json.dumps({"instruction": m[0]["content"], "output": m[-1]["content"], "id": str(i)}, ensure_ascii=False) + "\n")
    Curator(CuratorConfig(dataset=src, dedup="minhash", minhash_threshold=float(os.environ.get("MH", "0.97")), clean=True,
                          clean_transforms={"strip_html": False, "collapse_whitespace": False}, secrets_gate=True,
                          min_tokens=10, max_tokens=6000, export_formats=["alpaca"], output_dir=os.path.join(out, split))).run()
    man = json.load(open(os.path.join(out, split, "manifest.json"), encoding="utf-8"))
    rej = [json.loads(l) for l in open(os.path.join(out, split, "rejected.jsonl"), encoding="utf-8")] if os.path.exists(os.path.join(out, split, "rejected.jsonl")) else []
    why = collections.Counter(str(x.get("rejection_reason") or x.get("metadata", {}).get("rejection_reason") or sorted(x.get("metadata", {}))[:3])[:80] for x in rej)
    rep[split] = dict(rows_in=len(rows), stage_counts=man.get("stage_counts"), rejected=len(rej), reasons=why.most_common(8))
    print(split, json.dumps(rep[split])[:900], flush=True)
json.dump(rep, open(os.path.join(out, "curatorkit_audit_v4b.json"), "w", encoding="utf-8"), indent=1)
