"""Report for the cost experiment (A) and the ask-the-advisor experiment (B). See scratch/cost_ask_plan.py.

usage: cost_ask_report.py RUN_DIR HOLDOUT_REPORT.json OUT.json
Cost = tokens used x list price (USD per million tokens, OpenRouter list prices on 2026-11-13). Where the provider
reported a cost itself, that figure is used. Advisor calls are priced at the advisor's rate and included.
"""
import collections, json, os, sys

run, allowed_src, out = sys.argv[1], sys.argv[2], sys.argv[3]
PRICE = {  # input, output, cache read, cache write
    "claude-opus-5-5": (4.0, 20.0, 0.2, 5.0), "claude-haiku-5-5": (0.1, 0.5, 0.01, 0.125),
    "anthropic/claude-haiku-4.5": (1.0, 5.0, 0.1, 1.25), "openai/gpt-4o-mini": (0.15, 0.6, 0.075, 0.15),
    "google/gemini-2.5-flash-lite": (0.1, 0.4, 0.01, 0.083), "meta-llama/llama-3.1-8b-instruct": (0.02, 0.04, 0.02, 0.02)}
ADV = "claude-opus-5-5"
allowed = collections.defaultdict(set, {k: set(v) for k, v in json.load(open(allowed_src, encoding="utf-8"))["allowed_files"].items()})
ignore = lambda f: "__pycache__" in f or f.endswith(".pyc") or ".pytest_cache" in f


def price(model, u):
    p = PRICE[model]
    return (u.get("input", 0) * p[0] + u.get("output", 0) * p[1] + u.get("cacheRead", 0) * p[2] + u.get("cacheWrite", 0) * p[3]) / 1e6


metas = []
sd = os.path.join(run, "sessions")
for d in sorted(os.listdir(sd)):
    p = os.path.join(sd, d, "meta.json")
    if os.path.exists(p):
        m = json.load(open(p, encoding="utf-8"))
        m["_dir"] = os.path.join(sd, d)
        metas.append(m)
statuses = collections.Counter((m["model"].split("/")[-1], m["status"]) for m in metas)
metas = [m for m in metas if m["status"] in ("ok", "max_turns", "timeout")]
for m in metas:
    if m["condition"] == "clean" and m.get("arm") in ("cheap", "big", "ask_alone"):
        allowed[m["task_id"]] |= {f for f in m["files_changed"] if not ignore(f)}

rows = []
for m in metas:
    model = m["model"].split("/", 1)[1]
    agent_cost, tok = 0.0, 0
    for line in open(os.path.join(m["_dir"], "events.jsonl"), encoding="utf-8"):
        e = json.loads(line)
        if e.get("type") == "message_end" and e["message"].get("role") == "assistant":
            u = e["message"].get("usage") or {}
            c = u.get("cost") or {}
            agent_cost += c["total"] if c.get("totalFromProvider") else price(model, u)
            tok += u.get("totalTokens", 0)
    recs = [r for r in (m.get("loop") or {}).get("records", []) if "score" in r]
    calls = [r for r in recs if r.get("advisor_called")]
    asks = (m.get("ask") or {}).get("asks", [])
    adv_cost = sum(price(ADV, r.get("advisor_usage") or {}) for r in calls) + sum(price(ADV, a.get("usage") or {}) for a in asks)
    extra = [f for f in m["files_changed"] if not ignore(f) and f not in allowed[m["task_id"]]]
    passed = bool(m["check"] and m["check"]["passed"])
    rows.append(dict(sid=m["session_id"], model=model.split("/")[-1], task=m["task_id"], cond=m["condition"], arm=m.get("arm"),
                     passed=passed, extra=len(extra), good=passed and not extra, agent_cost=agent_cost, adv_cost=adv_cost,
                     cost=agent_cost + adv_cost, tokens=tok, calls=len(calls), asks=len(asks), answered=sum(bool(a["reply"]) for a in asks),
                     notes=sum(i["kind"] == "advisor" for i in m["injected"]), turns=m["turns"], seconds=m["seconds"]))
mean = lambda x: sum(x) / len(x) if x else float("nan")
tables = {}
print("statuses:", dict(statuses))
print("\n== A. COST (Haiku 5.5 cheap agent, Opus 5.5 as advisor and as the expensive alternative); cost in US cents per task")
print(f"{'condition':12} {'arm':10} {'n':>3} {'good':>6} {'check':>6} {'files':>6} {'agent c':>8} {'advisor c':>9} {'total c':>8} {'c per good task':>15} {'calls':>6} {'secs':>5}")
for cond in ("habit_side", "habit_setup", "induced", "clean", "all"):
    for arm in ("cheap", "cheap_adv", "big"):
        rs = [r for r in rows if r["arm"] == arm and (cond == "all" or r["cond"] == cond or (cond == "induced" and r["cond"] != "clean"))]
        if not rs:
            continue
        g = sum(r["good"] for r in rs)
        e = dict(n=len(rs), good=g, check=sum(r["passed"] for r in rs), extra=round(mean([r["extra"] for r in rs]), 2),
                 agent_cents=round(100 * mean([r["agent_cost"] for r in rs]), 3), advisor_cents=round(100 * mean([r["adv_cost"] for r in rs]), 3),
                 total_cents=round(100 * mean([r["cost"] for r in rs]), 3), cents_per_good=round(100 * sum(r["cost"] for r in rs) / g, 3) if g else None,
                 calls=round(mean([r["calls"] for r in rs]), 2), seconds=round(mean([r["seconds"] for r in rs])))
        tables[f"A|{cond}|{arm}"] = e
        print(f"{cond:12} {arm:10} {e['n']:3} {g:3}/{len(rs):<2} {e['check']:3}/{len(rs):<2} {e['extra']:6.2f} {e['agent_cents']:8.3f} {e['advisor_cents']:9.3f} {e['total_cents']:8.3f} {str(e['cents_per_good']):>15} {e['calls']:6.2f} {e['seconds']:5}")
print("\n== B. ASKING (clean tasks). check passes | sessions that asked | questions per session | cents per task | cents per passed task")
for model in sorted({r["model"] for r in rows if (r["arm"] or "").startswith("ask_")}):
    for arm in ("ask_alone", "ask_may", "ask_must"):
        rs = [r for r in rows if r["model"] == model and r["arm"] == arm]
        if not rs:
            continue
        g = sum(r["passed"] for r in rs)
        e = dict(n=len(rs), passed=g, asked=sum(r["asks"] > 0 for r in rs), asks=round(mean([r["asks"] for r in rs]), 2),
                 cents=round(100 * mean([r["cost"] for r in rs]), 3), cents_per_pass=round(100 * sum(r["cost"] for r in rs) / g, 3) if g else None,
                 passed_when_asked=f"{sum(r['passed'] for r in rs if r['asks'])}/{sum(r['asks'] > 0 for r in rs)}",
                 passed_when_not=f"{sum(r['passed'] for r in rs if not r['asks'])}/{sum(r['asks'] == 0 for r in rs)}")
        tables[f"B|{model}|{arm}"] = e
        print(f"{model:26} {arm:10} pass {g:2}/{len(rs):<2} | asked {e['asked']:2}/{len(rs):<2} | {e['asks']:.2f} | {e['cents']:7.3f} | {str(e['cents_per_pass']):>8} | pass if asked {e['passed_when_asked']}, if not {e['passed_when_not']}")
print("\nper task, check passes alone -> may -> must (all five agents pooled):")
for t in sorted({r["task"] for r in rows}):
    f = lambda arm: [r["passed"] for r in rows if r["task"] == t and r["arm"] == arm]
    print(f"  {t:18} {sum(f('ask_alone'))}/{len(f('ask_alone'))} -> {sum(f('ask_may'))}/{len(f('ask_may'))} -> {sum(f('ask_must'))}/{len(f('ask_must'))}")
json.dump(dict(rows=rows, tables=tables, prices_usd_per_mtok=PRICE), open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
