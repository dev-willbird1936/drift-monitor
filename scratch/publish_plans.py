"""Copy the closed-loop session plans into the publishable artifacts, with task paths made relative.

usage: publish_plans.py PILOT_DIR OUT_DIR
batch.py joins each plan's task onto the tasks/ folder, so unseen tasks become ../tasks_holdout/<id>.
"""
import io, json, os, sys

pilot, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for src, dst in (("plan.jsonl", "plan_seen_tasks.jsonl"), ("plan_holdout.jsonl", "plan_unseen_tasks.jsonl"),
                 ("plan_arms.jsonl", "plan_comparison_arms.jsonl")):
    rows = [json.loads(l) for l in open(os.path.join(pilot, src), encoding="utf-8") if l.strip()]
    for r in rows:
        t = r["task"].replace(chr(92), "/")
        name = t.rsplit("/", 1)[-1]
        r["task"] = "../tasks_holdout/" + name if "/tasks_holdout/" in t else name
    io.open(os.path.join(out, dst), "w", encoding="utf-8", newline="\n").write("\n".join(json.dumps(r) for r in rows) + "\n")
    print(dst, len(rows), sorted({r["task"] for r in rows})[:2], "...")
