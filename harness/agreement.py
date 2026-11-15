"""Step-level agreement between two judge label files. usage: agreement.py SESSIONS_DIR tagA tagB"""
import json, os, sys, collections
root, a, b = sys.argv[1:4]
c = collections.Counter(); cat = collections.Counter(); n = 0
for d in os.listdir(root):
    pa, pb = (os.path.join(root, d, f"labels_{t}.json") for t in (a, b))
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    n += 1
    A = {s["step"]: s for s in json.load(open(pa))["steps"]}; B = {s["step"]: s for s in json.load(open(pb))["steps"]}
    for k in A:
        if k in B:
            c[(A[k]["label"], B[k]["label"])] += 1
            if A[k]["label"] == B[k]["label"] == "drift":
                cat[A[k]["category"] == B[k]["category"]] += 1
tot = sum(c.values()); agree = c[("drift", "drift")] + c[("on_course", "on_course")]
pa_d = (c[("drift", "drift")] + c[("drift", "on_course")]) / tot; pb_d = (c[("drift", "drift")] + c[("on_course", "drift")]) / tot
pe = pa_d * pb_d + (1 - pa_d) * (1 - pb_d)
print(f"sessions {n} steps {tot} agreement {agree/tot:.3f} kappa {(agree/tot-pe)/(1-pe):.3f}")
print(dict(c)); print("category agreement on joint drift:", dict(cat))
