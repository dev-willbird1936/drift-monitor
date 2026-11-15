"""Does showing the monitor the supervisor's notes stop it re-alerting on the repair work those notes asked for?

usage: loop_notes_replay.py OUT.json RUN_DIR [RUN_DIR ...] [--arm monitor] [--url http://127.0.0.1:8765]
Replays watched sessions twice per turn: as run (notes hidden from the monitor) and with each delivered note
listed under "Later user messages" marked as a supervisor note. No retraining; exploratory.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as C

arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
arm, url = arg("--arm", "monitor"), arg("--url", "http://127.0.0.1:8765")
out_path = sys.argv[1]
runs = [a for a in sys.argv[2:] if not a.startswith("--") and a not in (arm, url)]
res = []
for run in runs:
    sd = os.path.join(run, "sessions")
    for d in sorted(os.listdir(sd)):
        m = json.load(open(os.path.join(sd, d, "meta.json"), encoding="utf-8"))
        notes = {i["text"] for i in m["injected"] if i["kind"] == "advisor"}
        if (m.get("arm") or "none") != arm or not notes:
            continue
        hidden = {i["text"] for i in m["injected"] if i["kind"] == "nudge"}
        msgs, turns, seen_note = [], [], False
        for line in open(os.path.join(sd, d, "events.jsonl"), encoding="utf-8"):
            e = json.loads(line)
            if e.get("type") == "message_end":
                msgs.append(e)
            if e.get("type") != "turn_end":
                continue
            items = C.S.parse(msgs, hidden, notes)
            steps, ups, kept, omitted = C.observer_view(m["goal"], m["constraints"], items)
            if not steps:
                continue
            k = len(steps)
            shown = [it["content"] for it in items if it["kind"] == "advisor"]
            if not shown:
                continue  # before the first note the two views are identical
            ups2 = ups + [{"after_step": 0, "text": "[note from the user's supervising reviewer] " + t.split("\n", 1)[-1]} for t in shown]
            a = C.score(url, C.R.build_prompt(m["goal"], m["constraints"], ups, kept, C.TIER, k, omitted))["score"]
            b = C.score(url, C.R.build_prompt(m["goal"], m["constraints"], ups2, kept, C.TIER, k, omitted))["score"]
            turns.append(dict(step=k, hidden=a, shown=b))
        if turns:
            res.append(dict(task=m["task_id"], cond=m["condition"], turns=turns))
            print(m["condition"].ljust(17), m["task_id"].ljust(22), [(round(t["hidden"]), round(t["shown"])) for t in turns], flush=True)
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
