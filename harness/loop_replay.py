"""Replay recorded sessions through the monitor with no intervention, one score per turn.

usage: loop_replay.py RUN_DIR OUT.json [--arm none] [--url http://127.0.0.1:8765]
Gives the alert rate the monitor would have had on sessions it never touched, and lets detection be scored
against objective outcomes (files left behind, failed check) instead of model-written labels.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as C

run, out = sys.argv[1], sys.argv[2]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
arm, url = arg("--arm", "none"), arg("--url", "http://127.0.0.1:8765")
res = {}
sd = os.path.join(run, "sessions")
for d in sorted(os.listdir(sd)):
    mp = os.path.join(sd, d, "meta.json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp, encoding="utf-8"))
    if (m.get("arm") or "none") != arm:
        continue
    hidden = {i["text"] for i in m["injected"] if i["kind"] == "nudge"}
    msgs, scores = [], []
    for line in open(os.path.join(sd, d, "events.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") == "message_end":
            msgs.append(e)
        if e.get("type") == "turn_end":
            items = C.S.parse(msgs, hidden)
            steps, ups, kept, omitted = C.observer_view(m["goal"], m["constraints"], items)
            if steps:
                k = len(steps)
                scores.append(C.score(url, C.R.build_prompt(m["goal"], m["constraints"], ups, kept, C.TIER, k, omitted))["score"])
    res[m["session_id"]] = dict(task=m["task_id"], cond=m["condition"], scores=scores)
    print(m["condition"].ljust(17), m["task_id"].ljust(22), [round(s) for s in scores], flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
