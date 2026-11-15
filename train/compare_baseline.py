"""Head-to-head: the 3.35B monitor versus frontier models doing the same monitoring job.

usage: compare_baseline.py OURS_JSON BASELINE_JSON [BASELINE_JSON ...] [--threshold 40] [--out OUT.json]

Both sides are restricted to the rows they have in common, so the comparison is on identical inputs with
identical evidence. Raw threshold-40 numbers favour whichever model happens to sit at a convenient operating
point, so this also compares at matched operating points: hold recall equal and compare false alarms, then
hold false alarms equal and compare recall. That is the fair way to ask "could the small model replace the
API call".
"""
import json, math, statistics, sys


def auroc(labels, scores):
    pos = sum(1 for l in labels if l)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rpos = sum(r for r, l in zip(ranks, labels) if l)
    return (rpos - pos * (pos + 1) / 2) / (pos * neg)


def at(rows, thr):
    tp = sum(1 for r in rows if r["pred"] >= thr and r["true"] >= 40)
    fp = sum(1 for r in rows if r["pred"] >= thr and r["true"] < 40)
    fn = sum(1 for r in rows if r["pred"] < thr and r["true"] >= 40)
    tn = sum(1 for r in rows if r["pred"] < thr and r["true"] < 40)
    return dict(thr=thr,
                recall=(tp / (tp + fn) if tp + fn else 0.0),
                precision=(tp / (tp + fp) if tp + fp else 0.0),
                false_alarm=(fp / (fp + tn) if fp + tn else 0.0),
                alert_rate=(tp + fp) / len(rows))


def sweep(rows):
    out = []
    for t in range(0, 101, 1):
        out.append(at(rows, float(t)))
    return out


def match_recall(rows, target):
    """Lowest false alarm achievable at or above a target recall."""
    best = None
    for p in sweep(rows):
        if p["recall"] >= target and (best is None or p["false_alarm"] < best["false_alarm"]):
            best = p
    return best


def match_false_alarm(rows, target):
    """Highest recall achievable at or below a target false-alarm rate."""
    best = None
    for p in sweep(rows):
        if p["false_alarm"] <= target and (best is None or p["recall"] > best["recall"]):
            best = p
    return best


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = {}
    for p in d["preds"]:
        rows[p["id"]] = dict(pred=float(p["pred"]), true=float(p["drift_score"]),
                             tier=p.get("tier"), secs=p.get("secs"))
    return d.get("summary", {}), rows


def describe(rows, label, thr):
    preds = [r["pred"] for r in rows]
    true = [r["true"] for r in rows]
    labs = [t >= 40 for t in true]
    a = auroc(labs, preds)
    base = at(rows, thr)
    return dict(model=label, n=len(rows),
                mae=round(sum(abs(p - t) for p, t in zip(preds, true)) / len(rows), 2),
                bias=round(sum(p - t for p, t in zip(preds, true)) / len(rows), 2),
                auroc=(round(a, 3) if a is not None else None),
                exact_digit=round(sum(1 for r in rows
                                      if min(9, int(r["pred"] // 10)) == min(9, int(r["true"] // 10)))
                                  / len(rows), 3),
                at_threshold={k: (round(v, 3) if isinstance(v, float) else v) for k, v in base.items()})


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    ours_path = sys.argv[1]
    thr = float(arg("--threshold", "40"))
    out_path = arg("--out", "comparison.json")
    # every positional .json after the first, excluding values that belong to a flag
    flag_values = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a.startswith("--")}
    base_paths = [a for a in sys.argv[2:]
                  if a.endswith(".json") and not a.startswith("--") and a not in flag_values]

    _, ours_all = load(ours_path)
    results, common_ids = {}, None
    baselines = {}
    for p in base_paths:
        s, rows = load(p)
        name = s.get("baseline_model", p)
        baselines[name] = (s, rows)
        common_ids = set(rows) if common_ids is None else (common_ids & set(rows))
    common_ids &= set(ours_all)
    ids = sorted(common_ids)
    print(f"{len(ids)} rows scored by every model, on identical inputs\n")

    ours = [dict(ours_all[i], id=i) for i in ids]
    results["drift-monitor-3.35B"] = describe(ours, "drift-monitor-3.35B (ours)", thr)
    for name, (s, rows) in baselines.items():
        r = [dict(rows[i], id=i) for i in ids]
        results[name] = describe(r, name, thr)
        lat = [x["secs"] for x in r if x.get("secs")]
        if lat:
            results[name]["latency_s_mean"] = round(statistics.mean(lat), 2)
            results[name]["latency_s_p90"] = round(sorted(lat)[int(len(lat) * 0.9)], 2)

    print(f"{'model':34} {'MAE':>6} {'bias':>7} {'AUROC':>7} {'exact':>7} {'recall':>7} {'prec':>6} {'FA':>6}")
    for k, v in results.items():
        t = v["at_threshold"]
        print(f"{v['model'][:34]:34} {v['mae']:>6} {v['bias']:>7} {str(v['auroc']):>7} "
              f"{v['exact_digit']:>7} {t['recall']:>7} {t['precision']:>6} {t['false_alarm']:>6}")

    # matched operating points, which is the comparison that actually means something
    print("\nmatched operating points (identical rows)")
    sets = {"drift-monitor-3.35B (ours)": ours}
    for name, (s, rows) in baselines.items():
        sets[name] = [dict(rows[i], id=i) for i in ids]

    print(f"\n  hold recall >= 0.70, compare false alarms:")
    for name, rws in sets.items():
        m = match_recall(rws, 0.70)
        print(f"    {name[:40]:40} " + (f"FA {m['false_alarm']:.3f} at thr {m['thr']:.0f} "
                                        f"(recall {m['recall']:.3f}, prec {m['precision']:.3f})"
                                        if m else "cannot reach 0.70 recall"))
    print(f"\n  hold false alarm <= 0.10, compare recall:")
    for name, rws in sets.items():
        m = match_false_alarm(rws, 0.10)
        print(f"    {name[:40]:40} " + (f"recall {m['recall']:.3f} at thr {m['thr']:.0f} "
                                        f"(FA {m['false_alarm']:.3f}, prec {m['precision']:.3f})"
                                        if m else "cannot reach FA <= 0.10"))

    matched = {}
    for name, rws in sets.items():
        mr, mf = match_recall(rws, 0.70), match_false_alarm(rws, 0.10)
        matched[name] = dict(
            at_recall_70=({k: round(v, 3) for k, v in mr.items()} if mr else None),
            at_false_alarm_10=({k: round(v, 3) for k, v in mf.items()} if mf else None))

    json.dump(dict(rows_compared=len(ids), threshold=thr, headline=results, matched_operating_points=matched),
              open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
