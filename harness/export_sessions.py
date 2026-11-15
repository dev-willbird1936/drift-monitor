"""Write scrubbed full session records (all items, hidden instructions marked). usage: export_sessions.py SESSIONS_DIR DATASET_DIR"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steps as S
from build_windows import scrub
root, out = sys.argv[1], sys.argv[2]
keep = {json.loads(l)["session_id"]: json.loads(l) for l in open(os.path.join(out, "sessions.jsonl"), encoding="utf-8")}
n = 0
with open(os.path.join(out, "sessions_full.jsonl"), "w", encoding="utf-8") as f:
    for d in sorted(os.listdir(root)):
        if d not in keep:
            continue
        meta, items = S.load(os.path.join(root, d))
        rec = dict(keep[d], goal=meta["goal"], constraints=meta["constraints"], hidden_system_prompt=meta.get("system_append"),
                   planted_files=meta.get("extra_files"), files_changed=[x for x in meta["files_changed"] if "__pycache__" not in x and ".pytest_cache" not in x],
                   items=[{"n": it["n"], "t": it["t"], "step": it["step"], "kind": it["kind"], "tool": it.get("tool"), "is_error": it.get("is_error"),
                           "content": scrub(it["content"])[:20000]} for it in items])
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
print(n, "sessions")
