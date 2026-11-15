"""Shortcut audit: how well do goal-blind and position-only baselines predict the label? usage: shortcut_check.py DATASET_DIR"""
import json, sys, os, collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d = sys.argv[1]
L = lambda sp: [w for w in (json.loads(l) for l in open(os.path.join(d, "data", sp + ".jsonl"), encoding="utf-8")) if w["tier"] == "actions" and w["clean"]]
tr, te, tc = L("train"), L("test"), L("test_category")
y = lambda W: np.array([w["label"] == "drift" for w in W])
last = lambda w: "\n".join(l for l in w["activity"].split("\n") if l.startswith(f"[step {w['step']}]"))
views = {"latest step only (goal-blind)": last, "all activity (goal-blind)": lambda w: w["activity"],
         "goal + updates + activity": lambda w: w["goal"] + "\n" + w["user_updates"] + "\n" + w["activity"]}
res = {}
for name, f in views.items():
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    m = LogisticRegression(max_iter=2000, C=3, class_weight="balanced").fit(v.fit_transform([f(w) for w in tr]), y(tr))
    res[name] = {"test": round(roc_auc_score(y(te), m.predict_proba(v.transform([f(w) for w in te]))[:, 1]), 3),
                 "test_category": round(roc_auc_score(y(tc), m.predict_proba(v.transform([f(w) for w in tc]))[:, 1]), 3)}
    if name.startswith("latest"):
        P = m.predict_proba(v.transform([f(w) for w in te]))[:, 1]
        by = collections.defaultdict(list)
        for w, p in zip(te, P):
            if w["label"] == "drift": by[w["category"]].append(p > 0.5)
        res["goal-blind recall by drift type (test)"] = {k: round(float(np.mean(x)), 2) for k, x in by.items()}
        neg = [p > 0.5 for w, p in zip(te, P) if w["label"] == "on_course"]
        res["goal-blind false alarm rate (test)"] = round(float(np.mean(neg)), 3)
pos = lambda W: np.array([[w["step"], w["step"] / w["n_steps"]] for w in W])
m = LogisticRegression().fit(pos(tr), y(tr))
res["step position only"] = {"test": round(roc_auc_score(y(te), m.predict_proba(pos(te))[:, 1]), 3)}
json.dump(res, open(os.path.join(d, "shortcut_audit.json"), "w"), indent=1); print(json.dumps(res, indent=1))
