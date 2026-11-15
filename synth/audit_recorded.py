"""Re-score real recorded sessions under the v4 rubric, to separate two explanations for the transfer gap.

usage: audit_recorded.py V3_DIR OUT_JSON --model provider/id [--thinking L] [--count N] [--workers N]

The tuned model scores real recorded sessions about 15 points higher than their stored v3 labels. Either the
model is wrong on real traffic, or the v3 labels were made under a more lenient rubric than the v4 one the
model was trained on. This asks an independent model to score the same real sessions under the v4 rubric and
compares all three: v3 label, independent v4-rubric judge, and the tuned model.
"""
import collections, json, math, os, random, re, statistics, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import render as R
from audit import PROMPT, body_of
from generate import RUBRIC, call_model


def to_session(rec):
    """Group the recorded item stream into the step shape render.py and audit.py expect."""
    by, order = {}, []
    for it in rec.get("items") or []:
        k = it.get("step")
        if not k:
            continue
        if k not in by:
            by[k] = dict(thinking="", say="", calls=[], results=[])
            order.append(k)
        b = by[k]
        if it["kind"] == "thinking":
            b["thinking"] += (" " if b["thinking"] else "") + it["content"]
        elif it["kind"] == "text":
            b["say"] += (" " if b["say"] else "") + it["content"]
        elif it["kind"] == "tool_call":
            try:
                args = json.loads(it["content"])
            except Exception:
                args = {"raw": it["content"][:200]}
            b["calls"].append(dict(tool=it.get("tool"), args=args))
        elif it["kind"] == "tool_result":
            b["results"].append(dict(text=it["content"], is_error=bool(it.get("is_error"))))
    order.sort()
    steps = [dict(n=i + 1, thinking_summary=(by[k]["thinking"] or "")[:90], **by[k]) for i, k in enumerate(order)]
    users = [it for it in (rec.get("items") or []) if it["kind"] == "user"]
    return dict(goal=rec.get("goal", ""), constraints=rec.get("constraints") or [],
                updates=[{"after_step": max(0, (it.get("step") or 1) - 1), "text": it["content"]} for it in users[1:]],
                steps=steps)


def pearson(a, b):
    if len(a) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return None if va == 0 or vb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def stats(x, y, xn, yn):
    d = [abs(p - q) for p, q in zip(x, y)]
    return {f"{xn}_mean": round(statistics.mean(x), 1), f"{yn}_mean": round(statistics.mean(y), 1),
            "mean_abs_diff": round(statistics.mean(d), 1), "median_abs_diff": round(statistics.median(d), 1),
            f"{yn}_minus_{xn}": round(statistics.mean(y) - statistics.mean(x), 1),
            "pearson": (round(pearson(x, y), 3) if pearson(x, y) is not None else None),
            "within_one_digit": round(sum(abs(int(min(9, p // 10)) - int(min(9, q // 10))) <= 1
                                          for p, q in zip(x, y)) / len(x), 3), "n": len(x)}


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    v3, out = sys.argv[1], sys.argv[2]
    model, thinking = arg("--model", "claude-bridge-1/claude-sonnet-5-5"), arg("--thinking", "high")
    count, workers = int(arg("--count", "50")), int(arg("--workers", "6"))
    preds_path = arg("--preds")

    recs = []
    for line in open(os.path.join(v3, "sessions_full.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("drift_scores") and (r.get("items") or []):
            recs.append(r)
    recs = random.Random(11).sample(recs, min(count, len(recs)))
    print(len(recs), "real sessions to re-score under the v4 rubric with", model, flush=True)

    def one(rec):
        sess = to_session(rec)
        if not sess["steps"]:
            return None
        try:
            txt = call_model(model, thinking, PROMPT.format(rubric=RUBRIC, body=body_of(sess)),
                             tempfile.mkdtemp(prefix="drift-rec-"))
            got = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))["steps"]
        except Exception:
            return None
        old = rec["drift_scores"]
        rows = []
        for s in got:
            n = s.get("n")
            if isinstance(n, int) and 1 <= n <= len(old):
                rows.append(dict(session=rec["session_id"], condition=rec.get("condition"),
                                 category=rec.get("category"), n=n,
                                 v3=float(old[n - 1]), v4judge=float(s["state_drift"])))
        return rows

    allrows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, recs):
            if r:
                allrows += r
    if not allrows:
        print("no rows"); return

    res = {"judge": model,
           "v3_label_vs_v4_rubric_judge": stats([r["v3"] for r in allrows], [r["v4judge"] for r in allrows], "v3", "v4judge")}

    # if the tuned model's predictions are supplied, add the three-way comparison
    if preds_path and os.path.exists(preds_path):
        pred = {}
        for p in json.load(open(preds_path, encoding="utf-8"))["preds"]:
            sid, rest = p["id"].split("#s", 1)
            step = int(rest.split("#")[0])
            pred.setdefault((sid, step), []).append(p["pred"])
        trip = [(r["v3"], r["v4judge"], statistics.mean(pred[(r["session"], r["n"])]))
                for r in allrows if (r["session"], r["n"]) in pred]
        if trip:
            res["three_way_n"] = len(trip)
            res["v3_vs_model"] = stats([t[0] for t in trip], [t[2] for t in trip], "v3", "model")
            res["v4judge_vs_model"] = stats([t[1] for t in trip], [t[2] for t in trip], "v4judge", "model")

    byc = collections.defaultdict(list)
    for r in allrows:
        byc[r["condition"]].append(r["v4judge"] - r["v3"])
    res["v4judge_minus_v3_by_condition"] = {k: dict(steps=len(v), mean_gap=round(statistics.mean(v), 1))
                                            for k, v in sorted(byc.items(), key=lambda kv: -statistics.mean(kv[1]))}
    json.dump(dict(summary=res, rows=allrows), open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "v4judge_minus_v3_by_condition"}, indent=1))
    print("\nhow much higher the v4 rubric scores each recorded condition:")
    for k, v in list(res["v4judge_minus_v3_by_condition"].items())[:10]:
        print(f"  {str(k):18} {v['mean_gap']:+6.1f}  ({v['steps']} steps)")


if __name__ == "__main__":
    main()
