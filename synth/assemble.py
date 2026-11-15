"""Assemble dataset v4: synthetic sessions at six evidence tiers, plus the recorded Pi sessions.

usage: assemble.py OUT_DIR --synth DIR [--recorded DATASET_V3_DIR] [--tiers-per-step 3]
"""
import collections, hashlib, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render as R
import taxonomy as T

# Whole work categories held out, to test generalisation to kinds of work never trained on.
HELD_OUT_CATEGORIES = {"hr-recruit", "creative-prod", "learning", "finance-admin"}
# Tier sampling weights: thin-evidence tiers are the realistic common case, so they get real weight.
TIER_WEIGHTS = {1: 0.9, 2: 1.0, 3: 1.4, 4: 1.0, 5: 1.6, 6: 1.1}


def h(s):
    return int(hashlib.sha1(str(s).encode()).hexdigest()[:8], 16)


def split_for(session_id, category):
    if category in HELD_OUT_CATEGORIES:
        return "test_category"
    r = h(session_id) % 100
    return "test" if r < 12 else ("val" if r < 18 else "train")


def pick_steps(n, rng, cap):
    """Sample at most `cap` step positions, always keeping the first and last."""
    ks = list(range(1, n + 1))
    if n <= cap:
        return ks
    keep = {1, n}
    keep |= set(rng.sample([k for k in ks if k not in keep], cap - len(keep)))
    return sorted(keep)


def from_synth(path, rng, tiers_per_step, cap=6):
    d = json.load(open(path, encoding="utf-8"))
    spec = d["spec"]
    steps = d["steps"]
    sid = spec["id"]
    split = split_for(sid, spec["category"])
    ups_all = d.get("updates") or []
    out = []
    for k in pick_steps(len(steps), rng, cap):
        upto = steps[:k]
        kept, omitted = R.budget_steps(upto, k)
        ups = [u for u in ups_all if u.get("after_step", 0) < k]
        s = steps[k - 1]
        score = float(s["state_drift"])
        tiers = rng.sample(list(TIER_WEIGHTS), k=len(TIER_WEIGHTS))
        tiers = sorted(tiers, key=lambda t: -TIER_WEIGHTS[t] * rng.random())[:tiers_per_step]
        for t in tiers:
            if t == 1 and not any(x.get("say") for x in kept):
                continue  # nothing visible to judge
            prompt = R.build_prompt(d["goal"], d.get("constraints"), ups, kept, t, k, omitted)
            out.append(dict(
                id=f"{sid}#s{k}#t{t}", session_id=sid, source="synthetic", step=k, n_steps=len(steps),
                tier=t, tier_name=R.TIERS[t], prompt=prompt, completion=str(min(9, int(score // 10))),
                drift_score=round(score), step_score=round(float(s["step_drift"])),
                advise=bool(s["advise"]), help=s.get("help"), why=s.get("why"),
                mode=spec["mode"], is_drift=spec["mode"] in T.DRIFT_MODES, task_category=spec["category"],
                band=T.CATEGORIES[spec["category"]]["band"], difficulty=spec["difficulty"],
                agent_tier=spec["agent_tier"], generator=spec.get("generator"), split=split))
    return out


def recorded_split(session_id, train_share):
    """Session-level split of the real corpus: no session appears in both train and test."""
    return "train" if (h("rec|" + str(session_id)) % 100) < train_share * 100 else "test_recorded"


def from_recorded(v3_dir, rng, tiers_per_step, train_share=0.0):
    """Recorded Pi sessions: real agents, real tools. Kept as a transfer test set."""
    path = os.path.join(v3_dir, "sessions_full.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        items = rec.get("items") or []
        scores = rec.get("drift_scores") or []
        if not scores:
            continue
        # group recorded items into steps
        by = collections.defaultdict(lambda: dict(thinking="", say="", calls=[], results=[]))
        order = []
        for it in items:
            k = it.get("step")
            if not k:
                continue
            if k not in by:
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
        order = sorted(order)
        steps = [dict(n=i + 1, **by[k]) for i, k in enumerate(order)]
        ups = [it for it in items if it["kind"] == "user"]
        ups_all = [{"after_step": max(0, (it.get("step") or 1) - 1), "text": it["content"]} for it in ups[1:]]
        sid = rec["session_id"]
        for i, k in enumerate(range(1, min(len(steps), len(scores)) + 1)):
            kept, omitted = R.budget_steps(steps[:k], k)
            score = float(scores[i])
            # recorded sessions have no provider summary, so tier 3 is not available for them
            pool = [t for t in TIER_WEIGHTS if t != 3]
            tiers = sorted(pool, key=lambda t: -TIER_WEIGHTS[t] * rng.random())[:tiers_per_step]
            for t in tiers:
                if t == 1 and not any(x.get("say") for x in kept):
                    continue
                prompt = R.build_prompt(rec.get("goal", ""), rec.get("constraints"), ups_all, kept, t, k, omitted)
                out.append(dict(
                    id=f"{sid}#s{k}#t{t}", session_id=sid, source="recorded", step=k, n_steps=len(steps),
                    tier=t, tier_name=R.TIERS[t], prompt=prompt, completion=str(min(9, int(score // 10))),
                    drift_score=round(score), step_score=round(score), advise=bool((rec.get("advise") or [False] * 99)[i]),
                    help=None, why=None, mode=rec.get("condition"), is_drift=score >= 40,
                    task_category=rec.get("category"), band="technical", difficulty="moderate",
                    agent_tier="mid", generator=rec.get("model"), split=recorded_split(sid, train_share)))
    return out


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    out_dir = sys.argv[1]
    synth_dir = arg("--synth")
    v3 = arg("--recorded")
    tps = int(arg("--tiers-per-step", "2"))
    rec_share = float(arg("--recorded-train-share", "0.0"))
    cap = int(arg("--steps-per-session", "6"))
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    rng = random.Random(4)

    rows = []
    files = sorted(f for f in os.listdir(synth_dir) if f.endswith(".json"))
    unreadable = []
    for i, f in enumerate(files):
        try:
            rows += from_synth(os.path.join(synth_dir, f), rng, tps, cap)
        except Exception as e:
            unreadable.append(f)
            print("skip", f, str(e)[:80])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} synthetic sessions -> {len(rows)} rows", flush=True)
    n_syn = len(rows)
    if v3:
        rows += from_recorded(v3, rng, tps, rec_share)
    print(f"{len(files)} synthetic sessions -> {n_syn} rows; recorded -> {len(rows)-n_syn} rows", flush=True)

    # Cross-split prompt dedup. The recorded corpus contains repeated runs of the same task, so an
    # identical prompt could land in train and in a test split. Training wins; the duplicate is dropped
    # from the evaluation side, because a test row the model has literally already seen is not a test.
    by = collections.defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    train_prompts = {hashlib.sha1(r["prompt"].encode()).hexdigest() for r in by.get("train", [])}
    dropped = collections.Counter()
    for sp in list(by):
        if sp == "train":
            continue
        keep = []
        for r in by[sp]:
            if hashlib.sha1(r["prompt"].encode()).hexdigest() in train_prompts:
                dropped[sp] += 1
            else:
                keep.append(r)
        by[sp] = keep
    if dropped:
        print("dropped as duplicates of a training prompt:", dict(dropped), flush=True)
    rows = [r for sp in by for r in by[sp]]
    # count sessions that made it into the dataset, not files on disk: truncated files are skipped above
    stats = {"sessions_synthetic": len(files) - len(unreadable), "synthetic_files_unreadable": unreadable, "rows_total": len(rows),
             "recorded_train_share": rec_share,
             "cross_split_duplicates_dropped": dict(dropped), "splits": {}}
    for sp, rs in by.items():
        with open(os.path.join(out_dir, "data", sp + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["splits"][sp] = dict(
            rows=len(rs), sessions=len({r["session_id"] for r in rs}),
            drift_rows=sum(r["is_drift"] for r in rs),
            digits=dict(sorted(collections.Counter(r["completion"] for r in rs).items())),
            tiers=dict(sorted(collections.Counter(r["tier"] for r in rs).items())))
    tr = by.get("train", [])
    stats["train_by_mode"] = dict(collections.Counter(r["mode"] for r in tr).most_common())
    stats["train_by_category"] = dict(collections.Counter(r["task_category"] for r in tr).most_common())
    stats["train_by_band"] = dict(collections.Counter(r["band"] for r in tr))
    stats["train_by_agent_tier"] = dict(collections.Counter(r["agent_tier"] for r in tr))
    stats["train_by_help"] = dict(collections.Counter(str(r["help"]) for r in tr))
    stats["train_by_source"] = dict(collections.Counter(r["source"] for r in tr))
    stats["prompt_chars_p99"] = sorted(len(r["prompt"]) for r in rows)[int(len(rows) * 0.99)]
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "train_by_mode"}, indent=1)[:1800])


if __name__ == "__main__":
    main()
