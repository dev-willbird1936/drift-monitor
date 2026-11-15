"""Run a plan of sessions with per-provider concurrency. Resumable: finished plan keys are skipped.

usage: batch.py PLAN.jsonl OUT_ROOT [--per-provider N]
plan line: {"key": "...", "task": "sw-slugify", "condition": "clean", "model": "xai/grok-4.7", "thinking": "low"}
"""
import json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.join(HERE, "..", "tasks")
plan_path, out = sys.argv[1], sys.argv[2]
per = int(sys.argv[sys.argv.index("--per-provider") + 1]) if "--per-provider" in sys.argv else 3
os.makedirs(out, exist_ok=True)
res_path = os.path.join(out, "results.jsonl")
done = set()
if os.path.exists(res_path):
    for l in open(res_path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("status") in ("ok", "max_turns", "timeout"):
            done.add(r["key"])
plan = [json.loads(l) for l in open(plan_path, encoding="utf-8") if l.strip()]
todo = [p for p in plan if p["key"] not in done]
print(f"{len(plan)} planned, {len(done)} done, {len(todo)} to run", flush=True)
sems, lock = {}, threading.Lock()


def run(p):
    prov = p["model"].split("/")[0]
    with lock:
        sem = sems.setdefault(prov, threading.Semaphore(per))
    with sem:
        cmd = [sys.executable, os.path.join(HERE, "run_session.py"), "--task", os.path.join(TASKS, p["task"]),
               "--condition", p["condition"], "--model", p["model"], "--thinking", p["thinking"], "--out", os.path.join(out, "sessions")] + list(p.get("extra") or [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            info = json.loads(last) if last.startswith("{") else {"status": "driver_error", "err": (r.stderr or r.stdout)[-400:]}
        except Exception as e:
            info = {"status": "driver_error", "err": str(e)[:400]}
        info.update(key=p["key"], model=p["model"], thinking=p["thinking"])
        with lock:
            open(res_path, "a", encoding="utf-8").write(json.dumps(info) + "\n")
            print(json.dumps(info)[:300], flush=True)


with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(run, todo))
