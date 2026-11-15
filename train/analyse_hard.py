"""Analyse the hard-scenario run: trigger policies, and a latency sweep over prompt length.

usage: analyse_hard.py HARD_JSON [--latency ADAPTER_DIR --tokenizer DIR --base MODEL]

Two questions. First, the monitor emits a score per step but the advisor decision can read the history: is
the instantaneous score the right trigger, or should it be the running maximum? Two of the misses were steps
where the score fired at the breach and then decayed while the breach was still unresolved, which is exactly
what a running maximum would catch. Second, what does a score actually cost in wall time as prompts grow.
"""
import json, statistics, sys


def policies(scen):
    """Compare trigger policies over the per-step timelines."""
    out = {}
    for name, fn in (
        ("instantaneous", lambda t: t[-1] >= 40),
        ("running_max", lambda t: max(t) >= 40),
        ("max_last3", lambda t: max(t[-3:]) >= 40),
        ("two_consecutive", lambda t: any(t[i] >= 40 and t[i + 1] >= 40 for i in range(len(t) - 1))),
        ("instantaneous_30", lambda t: t[-1] >= 30),
    ):
        ok = fa = miss = 0
        fa_names, miss_names = [], []
        for s in scen:
            fires = fn(s["timeline"])
            if fires == s["drift"]:
                ok += 1
            elif fires:
                fa += 1; fa_names.append(s["name"][:38])
            else:
                miss += 1; miss_names.append(s["name"][:38])
        out[name] = dict(correct=ok, total=len(scen), false_alarms=fa, missed=miss,
                         false_alarm_names=fa_names, missed_names=miss_names)
    return out


def main():
    path = sys.argv[1]
    d = json.load(open(path, encoding="utf-8"))
    scen, summ = d["scenarios"], d["summary"]

    print(f"{len(scen)} hard novel scenarios: {sum(s['drift'] for s in scen)} drift, "
          f"{sum(1 for s in scen if not s['drift'])} on-course\n")
    print("trigger policy comparison (threshold 40 unless stated):")
    pol = policies(scen)
    print(f"  {'policy':18} {'correct':>9} {'false':>6} {'missed':>7}")
    for k, v in pol.items():
        print(f"  {k:18} {v['correct']:>4}/{v['total']:<4} {v['false_alarms']:>6} {v['missed']:>7}")
    best = max(pol.items(), key=lambda kv: (kv[1]["correct"], -kv[1]["missed"]))
    print(f"\nbest: {best[0]} at {best[1]['correct']}/{best[1]['total']}")
    if best[1]["false_alarm_names"]:
        print("  its false alarms:", "; ".join(best[1]["false_alarm_names"]))
    if best[1]["missed_names"]:
        print("  its misses:", "; ".join(best[1]["missed_names"]))

    # decay: cases where the score peaked above the threshold then fell back under it
    print("\nscores that peaked over 40 then decayed below it:")
    any_decay = False
    for s in scen:
        t = s["timeline"]
        if max(t) >= 40 and t[-1] < 40:
            any_decay = True
            print(f"  {'drift ' if s['drift'] else 'course'} {s['group']:12} peak {max(t):>3} -> final {t[-1]:>3}   {s['name'][:46]}")
    if not any_decay:
        print("  none")

    print("\nby evidence tier:")
    for tier in sorted({s["etier"] for s in scen}):
        rs = [s for s in scen if s["etier"] == tier]
        print(f"  tier {tier}: decisions {sum(x['decision_ok'] for x in rs)}/{len(rs)}  "
              f"mean spread {statistics.mean(x['spread'] for x in rs):.1f}")

    L = summ["latency"]
    print(f"\nlatency measured over {L['n_scores']} scores on the local GPU")
    print(f"  mean {L['mean_ms']}ms   p50 {L['p50_ms']}ms   p90 {L['p90_ms']}ms   p99 {L['p99_ms']}ms   max {L['max_ms']}ms")
    print(f"  throughput {L['scores_per_sec']} scores/sec sequential; prompt tokens mean "
          f"{L['mean_prompt_tokens']} max {L['max_prompt_tokens']}; model load {summ['load_secs']}s")

    # latency vs prompt length, from the per-score samples
    sm = d.get("latency_samples") or []
    if sm:
        buckets = {}
        for x in sm:
            b = (x["tokens"] // 250) * 250
            buckets.setdefault(b, []).append(x["secs"] * 1000)
        print("\n  latency by prompt length:")
        print(f"    {'tokens':>12} {'n':>4} {'mean ms':>8} {'p90 ms':>7}")
        for b in sorted(buckets):
            v = sorted(buckets[b])
            print(f"    {str(b)+'-'+str(b+249):>12} {len(v):>4} {statistics.mean(v):>8.0f} {v[int(len(v)*0.9)]:>7.0f}")

    json.dump(dict(policies=pol, latency=L), open(path.replace(".json", "_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
