"""CuratorKIT pass over the clean train/val examples: clean, minhash dedup, secrets gate, trainer-ready export.
usage: run_curatorkit.py DATASET_DIR
"""
import json, os, sys
from curatorkit import Curator, CuratorConfig
d = sys.argv[1]
for split in ("train", "val"):
    src = os.path.join(d, "curated", f"{split}_input.jsonl")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    n = 0
    with open(src, "w", encoding="utf-8") as f:
        for l in open(os.path.join(d, "data", f"{split}.jsonl"), encoding="utf-8"):
            w = json.loads(l)
            if w["clean"]:
                f.write(json.dumps({"instruction": w["prompt"], "output": w["completion"], "id": w["id"]}, ensure_ascii=False) + "\n"); n += 1
    print(split, "input", n, flush=True)
    result = Curator(CuratorConfig(
        dataset=src, dedup="minhash", minhash_threshold=float(os.environ.get("MH", "0.97")), clean=True, clean_transforms={"strip_html": False, "collapse_whitespace": False},
        secrets_gate=True, min_tokens=10, max_tokens=6000,
        export_formats=["alpaca", "sharegpt"], output_dir=os.path.join(d, "curated", split),
    )).run()
    print(json.dumps(json.load(open(os.path.join(d, "curated", split, "manifest.json")))["stage_counts"]))
