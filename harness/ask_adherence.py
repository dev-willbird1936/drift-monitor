"""Judge how well an agent carried out the advisor's numbered instructions after asking for them.

usage: ask_adherence.py RUN_DIR OUT.json [--model claude-bridge-1/claude-sonnet-5-5]
For every session with an answered question, a separate model reads the instructions and what the agent did
afterwards, and counts instructions followed, partly followed and not followed. Model-judged, so secondary evidence.
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop as C

run, out = sys.argv[1], sys.argv[2]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "claude-bridge-1/claude-sonnet-5-5"
SYSTEM = "You audit whether an AI agent followed instructions. Output only a JSON object."
GUIDE = """An agent asked an advisor for help and received numbered instructions. Below are the instructions and everything the
agent did afterwards. For each numbered instruction decide: "followed" (done as instructed), "partial" (attempted, done
differently or incompletely), or "not_followed" (ignored or contradicted). Judge only from the record.
Return exactly: {"instructions": <count>, "followed": <count>, "partial": <count>, "not_followed": <count>, "note": "<max 20 words>"}"""


def one(d):
    m = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    asks = [a for a in (m.get("ask") or {}).get("asks", []) if a.get("reply")]
    if not asks or m["status"] not in ("ok", "max_turns", "timeout"):
        return None
    items = C.S.load(d)[1]
    first = next((i for i, it in enumerate(items) if it["kind"] == "advisor"), None)
    if first is None:
        return None
    after = [it for it in items[first + 1:] if it["kind"] != "advisor"]
    record = C.S.render(after, clip=900) or "(the agent did nothing further)"
    prompt = f"{GUIDE}\n\n### Instructions the agent received\n{asks[0]['reply']}\n\n### What the agent did afterwards\n{record[:14000]}\n\nReturn the JSON now."
    text, _ = C.call_model(model, "low", SYSTEM, prompt)
    g = re.search(r"\{.*\}", text, re.S)
    try:
        j = json.loads(g.group(0)) if g else None
    except ValueError:
        j = None
    return dict(sid=m["session_id"], model=m["model"].split("/")[-1], arm=m["arm"], task=m["task_id"],
                passed=bool(m["check"] and m["check"]["passed"]), judge=j)


dirs = [os.path.join(run, "sessions", d) for d in sorted(os.listdir(os.path.join(run, "sessions")))]
with ThreadPoolExecutor(4) as ex:
    res = [r for r in ex.map(one, dirs) if r]
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
by = {}
for r in res:
    j = r["judge"]
    if not j or not j.get("instructions"):
        continue
    b = by.setdefault(r["model"], dict(sessions=0, instr=0, followed=0, partial=0, not_followed=0, all_followed=0, passed=0))
    b["sessions"] += 1; b["instr"] += j["instructions"]; b["followed"] += j["followed"]; b["partial"] += j["partial"]
    b["not_followed"] += j["not_followed"]; b["all_followed"] += j["followed"] == j["instructions"]; b["passed"] += r["passed"]
print(len(res), "sessions judged;", sum(1 for r in res if not r["judge"]), "unparsed")
for k, b in sorted(by.items()):
    print(f"{k:26} sessions {b['sessions']:2} | instructions followed {b['followed']}/{b['instr']} ({100 * b['followed'] / b['instr']:.0f}%), partial {b['partial']}, not {b['not_followed']} | followed all: {b['all_followed']}/{b['sessions']} | check passed {b['passed']}/{b['sessions']}")
