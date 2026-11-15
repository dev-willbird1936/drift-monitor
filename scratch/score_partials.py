"""Add mid-thought drift scores to submission/space/replays.json, so the replay can show the score moving while the
agent is still reasoning, as the live mode does.

For every step with enough reasoning, the session is scored with the earlier steps plus the first 35% and 70% of this
step's reasoning, rendered at evidence tier 6 (a live stream that breaks off mid-thought), exactly as the live mode
sends it. Sessions are the ones scratch/build_replays.py picked.

usage: score_partials.py SCORE_SERVER_URL
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scratch"))
import build_replays as B  # also puts harness/, synth/ and the Space on the path
import closed_loop as CL
import render as R
import steps as S
from driftwatch import LocalServerScorer

srv = LocalServerScorer(sys.argv[1])
P = os.path.join(ROOT, "submission", "space", "replays.json")
reps = json.load(open(P, encoding="utf-8"))
R._stream_cut = lambda t: " ".join(str(t).split())  # the text streamed so far is used as is
FRACS = (0.35, 0.7)

for pick, rep in zip(B.PICKS, reps):
    run = pick["run"]
    sess = B.sessions(run)
    pair = B.auto_pair(run, pick["watched"].split(":")[1]) if pick["watched"].startswith("auto:") else None
    for arm in ("watched", "unwatched"):
        sid = (pair[0] if arm == "watched" else pair[1]) if pair else next(s for s in sess if s.endswith(pick[arm]))
        meta, items = S.load(f"{B.LP}/{run}/sessions/{sid}")
        goal, cons = meta["goal"], meta["constraints"]
        n = max((it.get("step") or 0) for it in items)
        partials = {}
        for k in range(1, n + 1):
            think = " ".join(it["content"] for it in items if it.get("step") == k and it["kind"] == "thinking")
            if len(think) < 120:
                continue
            prior = [it for it in items if (it.get("step") or 0) < k and it["kind"] not in ("hidden_user", "advisor")]
            out = []
            for f in FRACS:
                sub = prior + [dict(kind="thinking", step=k, content=think[:int(len(think) * f)])]
                _, up, kept, om = CL.observer_view(goal, cons, sub)
                s, _ = srv(R.build_prompt(goal, cons, up, kept, 6, k, om))
                out.append(dict(frac=f, score=round(float(s), 1)))
            partials[str(k)] = out
        rep[arm]["partials"] = partials
        final = {t["step"]: t["score"] for t in rep[arm]["trajectory"]}
        print(rep["short"]["title"][:22], arm[:5], " ".join(f"{k}:{[p['score'] for p in v]}->{final.get(int(k))}" for k, v in partials.items()))
open(P, "w", encoding="utf-8").write(json.dumps(reps, ensure_ascii=False, indent=1))
print("wrote", P)
