"""Run the Space's live agent once against the local scoring server. usage: test_live_agent.py <drift kind> [task id] [advisor 0/1]"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "space"))
import live_agent as L
from driftwatch import LocalServerScorer

kind = sys.argv[1] if len(sys.argv) > 1 else "side"
tid = sys.argv[2] if len(sys.argv) > 2 else "ho-recipe-scale"
adv = (sys.argv[3] if len(sys.argv) > 3 else "1") == "1"
key = os.environ["OPENROUTER_API_KEY"]
tasks = {t["id"]: t for t in json.load(open(os.path.join(os.path.dirname(L.__file__), "tasks.json"), encoding="utf-8"))}
t = tasks[tid]
last = None
for st in L.run(key, LocalServerScorer("http://127.0.0.1:8777"), t["goal"], t["constraints"], t["files"], t["allowed"], kind, t, "",
                "google/gemini-3.1-flash-lite", "high", "anthropic/claude-haiku-5.5", adv, 40):
    last = st
for e in last["events"]:
    if e["kind"] == "step":
        print(f"[step {e['step']}] think={e['think'][:120]!r} say={e['say'][:120]!r}")
        for c in e["calls"]:
            print(f"     {c['tool']} {c['args'][:100]} -> {c['result'][:80]!r}")
    else:
        print(f"[{e['kind']}] {e['text'][:300]!r}")
print("scores", [(s["step"], s["score"]) for s in last["scores"]])
print("status", last["status"], "cost $%.4f" % last["cost"], "advisor calls", last["advisor_calls"])
print("outcome", last["outcome"])
