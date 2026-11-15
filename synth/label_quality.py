"""Combine the independent audits into one label-quality report.

usage: label_quality.py OUT_JSON AUDIT_JSON [AUDIT_JSON ...]

Each audit holds a different model's blind re-scoring of the same synthetic sessions. Agreement between the
generator and both auditors, and between the auditors themselves, is the evidence that the labels mean
something. Per-mode spread shows which patterns are intrinsically contested.
"""
import collections, json, math, statistics, sys


def pearson(a, b):
    if len(a) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((x - mb) ** 2 for x in b))
    return None if va == 0 or vb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def pair_stats(x, y):
    d = [abs(p - q) for p, q in zip(x, y)]
    return dict(n=len(x), mean_abs_diff=round(statistics.mean(d), 1), median_abs_diff=round(statistics.median(d), 1),
                pearson=(round(pearson(x, y), 3) if pearson(x, y) is not None else None),
                same_digit=round(sum(int(min(9, p // 10)) == int(min(9, q // 10)) for p, q in zip(x, y)) / len(x), 3),
                within_one_digit=round(sum(abs(int(min(9, p // 10)) - int(min(9, q // 10))) <= 1 for p, q in zip(x, y)) / len(x), 3))


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    audits = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        audits[d["summary"]["auditor"]] = d

    rep = {"auditors": {}, "generator_vs_auditor": {}, "auditor_vs_auditor": {}, "contested_modes": {}}
    # per auditor, against the generator
    index = {}
    for name, d in audits.items():
        rep["auditors"][name] = {k: v for k, v in d["summary"].items() if k != "worst_modes"}
        gen = [r["gen"] for r in d["rows"]]
        aud = [r["aud"] for r in d["rows"]]
        rep["generator_vs_auditor"][name] = pair_stats(gen, aud)
        rep["generator_vs_auditor"][name]["auditor_mean_minus_generator"] = round(statistics.mean(aud) - statistics.mean(gen), 1)
        rep["generator_vs_auditor"][name]["advise_agreement"] = d["summary"]["advise_agreement"]
        for r in d["rows"]:
            index.setdefault((r["session"], r["n"]), {})[name] = r

    # where two auditors scored the same step, compare them directly
    names = sorted(audits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xs = [(v[a]["aud"], v[b]["aud"]) for v in index.values() if a in v and b in v]
            if len(xs) >= 20:
                rep["auditor_vs_auditor"][f"{a} vs {b}"] = pair_stats([x for x, _ in xs], [y for _, y in xs])
            else:
                rep["auditor_vs_auditor"][f"{a} vs {b}"] = dict(overlapping_steps=len(xs), note="too few shared steps to compare")

    # which patterns the judges disagree about most, pooled over auditors
    bym = collections.defaultdict(list)
    for d in audits.values():
        for r in d["rows"]:
            bym[r["mode"]].append(abs(r["gen"] - r["aud"]))
    rep["contested_modes"] = {m: dict(steps=len(v), mean_abs_diff=round(statistics.mean(v), 1))
                              for m, v in sorted(bym.items(), key=lambda kv: -statistics.mean(kv[1]))}

    allgen, allaud = [], []
    for d in audits.values():
        allgen += [r["gen"] for r in d["rows"]]
        allaud += [r["aud"] for r in d["rows"]]
    rep["pooled"] = pair_stats(allgen, allaud)
    rep["pooled"]["generator_mean"] = round(statistics.mean(allgen), 1)
    rep["pooled"]["auditor_mean"] = round(statistics.mean(allaud), 1)

    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "contested_modes"}, indent=1))
    print("\nmost contested patterns:")
    for m, v in list(rep["contested_modes"].items())[:10]:
        print(f"  {m:20} {v['mean_abs_diff']:5.1f}  ({v['steps']} steps)")
    print("\nleast contested patterns:")
    for m, v in list(rep["contested_modes"].items())[-6:]:
        print(f"  {m:20} {v['mean_abs_diff']:5.1f}  ({v['steps']} steps)")


if __name__ == "__main__":
    main()
