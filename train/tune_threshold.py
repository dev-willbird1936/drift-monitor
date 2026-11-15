"""Choose the advisor-call threshold from predictions, and show what each choice costs.

usage: tune_threshold.py EVAL_JSON [EVAL_JSON ...] [--advisor-cost 1.0] [--miss-cost 8.0] [--cooldown 3]

The monitor emits a score; the decision is whether to spend a stronger model's call on a review. That is a
threshold, and the right threshold depends on the relative cost of a wasted review and a missed drift. This
sweeps every threshold and reports the operating point, so the number is chosen from data rather than taste.

A session-level view is included because in practice one alert per drifting session is enough: a cooldown
suppresses repeats, so what matters is whether a drifting session is caught at all, and how early.
"""
import collections, json, sys


def main():
    arg = lambda k, d: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d)
    paths = [a for a in sys.argv[1:] if a.endswith(".json")]
    advisor_cost = float(arg("--advisor-cost", 1.0))
    miss_cost = float(arg("--miss-cost", 8.0))

    rows = []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        split = d["summary"]["split"]
        for r in d["preds"]:
            r["split"] = split
            rows.append(r)
    if not rows:
        sys.exit("no predictions found")

    print(f"{len(rows)} predictions from {len(paths)} file(s)\n")
    print(f"{'thr':>4} {'recall':>7} {'prec':>6} {'falarm':>7} {'alerts%':>8} {'cost':>8}   {'sess_caught':>11} {'sess_false':>10}")

    # group by session for the session-level view
    by_sess = collections.defaultdict(list)
    for r in rows:
        sid = r["id"].split("#")[0]
        by_sess[sid].append(r)
    drift_sessions = {s for s, rs in by_sess.items() if any(x["drift_score"] >= 40 for x in rs)}
    clean_sessions = set(by_sess) - drift_sessions

    best = None
    for thr in range(10, 91, 5):
        tp = sum(1 for r in rows if r["pred"] >= thr and r["drift_score"] >= 40)
        fp = sum(1 for r in rows if r["pred"] >= thr and r["drift_score"] < 40)
        fn = sum(1 for r in rows if r["pred"] < thr and r["drift_score"] >= 40)
        tn = sum(1 for r in rows if r["pred"] < thr and r["drift_score"] < 40)
        recall = tp / (tp + fn) if tp + fn else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        falarm = fp / (fp + tn) if fp + tn else 0.0
        alerts = (tp + fp) / len(rows)
        cost = (fp * advisor_cost + fn * miss_cost) / len(rows)
        caught = sum(1 for s in drift_sessions if any(r["pred"] >= thr for r in by_sess[s]))
        sess_recall = caught / len(drift_sessions) if drift_sessions else 0.0
        sess_false = (sum(1 for s in clean_sessions if any(r["pred"] >= thr for r in by_sess[s]))
                      / len(clean_sessions) if clean_sessions else 0.0)
        print(f"{thr:>4} {recall:>7.3f} {prec:>6.3f} {falarm:>7.3f} {alerts:>8.1%} {cost:>8.3f}   "
              f"{sess_recall:>11.3f} {sess_false:>10.3f}")
        if best is None or cost < best[1]:
            best = (thr, cost, recall, prec, falarm, sess_recall, sess_false)

    thr, cost, recall, prec, falarm, sr, sf = best
    print(f"\nlowest expected cost at threshold {thr} "
          f"(advisor call costs {advisor_cost}, missed drift costs {miss_cost})")
    print(f"  step level:    recall {recall:.3f}  precision {prec:.3f}  false alarm {falarm:.3f}")
    print(f"  session level: {sr:.1%} of drifting sessions caught, "
          f"{sf:.1%} of clean sessions raise at least one alert")

    # how early the catch happens, at the chosen threshold
    delays = []
    for s in drift_sessions:
        rs = sorted(by_sess[s], key=lambda r: r["id"])
        first_true = next((i for i, r in enumerate(rs) if r["drift_score"] >= 40), None)
        first_pred = next((i for i, r in enumerate(rs) if r["pred"] >= thr), None)
        if first_true is not None and first_pred is not None:
            delays.append(first_pred - first_true)
        if delays:
            pass
    if delays:
        delays.sort()
        on_time = sum(1 for d in delays if d <= 0) / len(delays)
        print(f"  timing: median delay {delays[len(delays)//2]} scored steps after drift begins; "
              f"{on_time:.1%} flagged at or before the first drifting step")

    # per-split and per-tier at the chosen threshold
    for key in ("split", "tier"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r.get(key)].append(r)
        if len(groups) > 1:
            print(f"\n  by {key} at threshold {thr}:")
            for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
                tp = sum(1 for r in rs if r["pred"] >= thr and r["drift_score"] >= 40)
                fp = sum(1 for r in rs if r["pred"] >= thr and r["drift_score"] < 40)
                fn = sum(1 for r in rs if r["pred"] < thr and r["drift_score"] >= 40)
                tn = sum(1 for r in rs if r["pred"] < thr and r["drift_score"] < 40)
                rc = tp / (tp + fn) if tp + fn else 0.0
                pr = tp / (tp + fp) if tp + fp else 0.0
                fa = fp / (fp + tn) if fp + tn else 0.0
                print(f"    {str(g):16} n {len(rs):6}  recall {rc:.3f}  precision {pr:.3f}  false alarm {fa:.3f}")


if __name__ == "__main__":
    main()
