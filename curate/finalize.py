"""Merge CuratorKIT output with secrets-gate false positives (hash digests from task fixtures). usage: finalize.py DATASET_DIR"""
import json, os, re, sys, collections
d = sys.argv[1]
rep = {}
for split in ("train", "val"):
    c = os.path.join(d, "curated", split)
    keep = [json.loads(l) for l in open(os.path.join(c, "sft_alpaca.jsonl"), encoding="utf-8")]
    rej = [json.loads(l) for l in open(os.path.join(c, "rejected.jsonl"), encoding="utf-8")]
    restored, dropped, vals = [], [], collections.Counter()
    for x in rej:
        reason = json.dumps(x.get("metadata", {})) + str(x.get("rejection_reason", ""))
        text = x["instruction"] + (x.get("input") or "")
        digests = set(re.findall(r"(?<![0-9a-f])(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})(?![0-9a-f])", text))
        if "Hex High Entropy" in reason and digests and "Base64" not in reason:
            restored.append({"instruction": x["instruction"], "input": x.get("input") or "", "output": x["output"]}); vals.update(digests)
        else:
            dropped.append(reason[:200])
    final = keep + restored
    with open(os.path.join(d, "curated", f"{split}_final.jsonl"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]}, ensure_ascii=False) + "\n")
    rep[split] = {"curatorkit_kept": len(keep), "restored_hash_false_positives": len(restored), "dropped": len(dropped), "final": len(final),
                  "labels": dict(collections.Counter(r["output"] for r in final)), "distinct_digests": len(vals), "top_digests": vals.most_common(3),
                  "dropped_reasons": dropped[:5]}
json.dump(rep, open(os.path.join(d, "curated", "curation_report.json"), "w"), indent=1); print(json.dumps(rep, indent=1)[:2600])
