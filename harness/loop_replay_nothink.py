"""Replay recorded sessions through the monitor twice per turn: as recorded, and with the agent's thinking removed.

usage: loop_replay_nothink.py RUN_DIR OUT.json [--arm monitor_only] [--url http://127.0.0.1:8765]
Both passes use the same tier-5 prompt and the same sessions, so the score difference isolates what visible
reasoning contributes. Thinking items are dropped before observer_view, which is what steps.parse would yield
had the provider exposed no reasoning: a step that held only thinking disappears and later steps renumber.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as C

run, out = sys.argv[1], sys.argv[2]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
arm, url = arg("--arm", "monitor_only"), arg("--url", "http://127.0.0.1:8765")


def prompt_for(m, items):
    """The tier-5 monitor prompt for these items, or None when no agent step is visible."""
    steps, ups, kept, omitted = C.observer_view(m["goal"], m["constraints"], items)
    if not steps:
        return None, 0
    return C.R.build_prompt(m["goal"], m["constraints"], ups, kept, C.TIER, len(steps), omitted), len(steps)


res = {}
sd = os.path.join(run, "sessions")
for d in sorted(os.listdir(sd)):
    mp = os.path.join(sd, d, "meta.json")
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp, encoding="utf-8"))
    if (m.get("arm") or "none") != arm or m["status"] not in ("ok", "max_turns", "timeout"):
        continue
    hidden = {i["text"] for i in m["injected"] if i["kind"] == "nudge"}
    msgs, turns, turn = [], [], 0
    for line in open(os.path.join(sd, d, "events.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") == "message_end":
            msgs.append(e)
        if e.get("type") == "turn_end":
            turn += 1
            items = C.S.parse(msgs, hidden)
            p_rec, k_rec = prompt_for(m, items)
            if p_rec is None:
                continue
            p_no, k_no = prompt_for(m, [it for it in items if it["kind"] != "thinking"])
            turns.append(dict(
                turn=turn, step=k_rec, step_nothink=k_no,
                thinking_chars=sum(len(it["content"]) for it in items if it["kind"] == "thinking"),
                prompt_identical=p_rec == p_no,
                score_recorded=C.score(url, p_rec)["score"],
                score_nothink=C.score(url, p_no)["score"] if p_no is not None else None))
    res[m["session_id"]] = dict(task=m["task_id"], cond=m["condition"], model=m["model"], turns=turns)
    print(m["model"].split("/")[-1].ljust(16), m["condition"].ljust(12), m["task_id"].ljust(16),
          [round(t["score_recorded"]) for t in turns], "->",
          [None if t["score_nothink"] is None else round(t["score_nothink"]) for t in turns], flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
