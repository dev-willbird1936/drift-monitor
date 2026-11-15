"""Replay recorded sessions through the monitor at several evidence levels, each with its matching prompt header.

usage: loop_replay_views.py OUT.json RUN_DIR [RUN_DIR ...] [--url http://127.0.0.1:8765]

Views, scored at every turn of every session:
  replies   tier 1  the agent's chat replies only
  progress  tier 2  replies + which tool ran
  tools     tier 3  replies + tool calls and results, no reasoning at all
  summary   tier 3  the above + a short summary of the reasoning
  full      tier 5  the complete record as the harness received it

The summary is the provider's own when the visible thinking is already a short headline (under 90 characters,
as GPT models send); otherwise it is a proxy: the first sentence of the visible thinking, cut at 140 characters.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as C

arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
url = arg("--url", "http://127.0.0.1:8765")
out_path = sys.argv[1]
runs = [a for a in sys.argv[2:] if not a.startswith("--") and a != url]
VIEWS = ("replies", "progress", "tools", "summary", "full")


def summarise(think):
    think = think.strip()
    if len(think) < 90:
        return think.strip("* ")
    first = re.split(r"(?<=[.!?])\s", think, 1)[0]
    return first[:140]


def prompts(m, items):
    steps, ups, kept, omitted = C.observer_view(m["goal"], m["constraints"], items)
    if not steps:
        return None
    k = len(steps)
    bp = lambda st, tier: C.R.build_prompt(m["goal"], m["constraints"], ups, st, tier, k, omitted)
    bare = [dict(s, thinking="") for s in kept]
    summ = [dict(s, thinking="", thinking_summary=summarise(s["thinking"]) if s["thinking"] else "") for s in kept]
    return dict(replies=bp(kept, 1), progress=bp(kept, 2), tools=bp(bare, 3), summary=bp(summ, 3), full=bp(kept, 5))


res = {}
for run in runs:
    sd = os.path.join(run, "sessions")
    for d in sorted(os.listdir(sd)):
        mp = os.path.join(sd, d, "meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp, encoding="utf-8"))
        if m["status"] not in ("ok", "max_turns", "timeout"):
            continue
        hidden = {i["text"] for i in m["injected"] if i["kind"] == "nudge"}
        msgs, turns, cache = [], [], {}
        for line in open(os.path.join(sd, d, "events.jsonl"), encoding="utf-8"):
            e = json.loads(line)
            if e.get("type") == "message_end":
                msgs.append(e)
            if e.get("type") != "turn_end":
                continue
            items = C.S.parse(msgs, hidden)
            ps = prompts(m, items)
            if not ps:
                continue
            row = {}
            for v in VIEWS:
                if ps[v] not in cache:
                    cache[ps[v]] = C.score(url, ps[v])["score"]
                row[v] = cache[ps[v]]
            turns.append(row)
        think = [it["content"] for it in C.S.parse(msgs, hidden) if it["kind"] == "thinking"]
        res[m["session_id"]] = dict(task=m["task_id"], cond=m["condition"], model=m["model"], thinking_level=m["thinking"],
                                    files_changed=[f for f in m["files_changed"] if "__pycache__" not in f],
                                    check=bool(m["check"] and m["check"]["passed"]),
                                    thinking_blocks=len(think), thinking_chars=sum(map(len, think)), turns=turns)
        print(m["model"].split("/")[-1].ljust(18), m["condition"].ljust(12), m["task_id"].ljust(16),
              {v: max(round(t[v]) for t in turns) for v in VIEWS}, flush=True)
json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
print("wrote", out_path)
