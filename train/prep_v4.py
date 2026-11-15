"""Build the balanced fine-tuning set from dataset v4.

usage: prep_v4.py DATASET_V4_DIR OUT_DIR [--max-tokens 2000] [--cap-mult 2.2] [--train-cap N] [--val-cap 500]

v3 left 71% of its training rows on digit 0 and the resulting model under-predicted drift. Here every digit
class is capped at `cap-mult` times the median class size, which lifts the rare high-drift classes without
inventing them, and the surviving rows are kept proportional across evidence tiers.
"""
import collections, json, os, random, sys

from transformers import AutoTokenizer


def main():
    d, out = sys.argv[1], sys.argv[2]
    arg = lambda k, v: (sys.argv[sys.argv.index(k) + 1] if k in sys.argv else v)
    max_tok = int(arg("--max-tokens", 2000))
    cap_mult = float(arg("--cap-mult", 2.2))
    zero_share = float(arg("--zero-share", 0.35))
    train_cap = int(arg("--train-cap", 0))
    val_cap = int(arg("--val-cap", 500))
    tok = AutoTokenizer.from_pretrained(arg("--tokenizer", os.environ.get("DRIFT_TOKENIZER", "./tokenizer")))
    rng = random.Random(13)
    rep = {}

    for split, cap in (("train", train_cap), ("val", val_cap)):
        path = os.path.join(d, "data", split + ".jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        n0 = len(rows)

        keep = []
        for r in rows:
            text = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["completion"]}],
                tokenize=False)
            if len(tok(text, add_special_tokens=False).input_ids) <= max_tok:
                keep.append(r)

        by_digit = collections.defaultdict(list)
        for r in keep:
            by_digit[r["completion"]].append(r)
        sizes = sorted(len(v) for v in by_digit.values())
        median = sizes[len(sizes) // 2] if sizes else 0
        limit = max(1, int(median * cap_mult))

        sel = []
        for dig, rs in by_digit.items():
            # digit 0 is the bulk of real traffic; keep it the largest class but not the dominant one,
            # then tune the alert threshold on a validation split that keeps the natural distribution.
            cap_for = limit
            if dig == "0":
                others = sum(min(len(v), limit) for k, v in by_digit.items() if k != "0")
                cap_for = max(limit, int(others * zero_share / max(1e-6, 1 - zero_share)))
            if len(rs) <= cap_for:
                sel += rs
                continue
            limit_here = cap_for
            # when trimming a class, keep its evidence tiers proportional rather than whichever came first
            by_tier = collections.defaultdict(list)
            for r in rs:
                by_tier[r["tier"]].append(r)
            share = {t: len(v) / len(rs) for t, v in by_tier.items()}
            for t, v in by_tier.items():
                rng.shuffle(v)
                sel += v[:max(1, round(limit_here * share[t]))]

        rng.shuffle(sel)
        if cap:
            sel = sel[:cap]

        os.makedirs(os.path.join(out, split), exist_ok=True)
        with open(os.path.join(out, split, "sft.jsonl"), "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]}]}, ensure_ascii=False) + "\n")

        rep[split] = dict(
            rows_in=n0, within_token_cap=len(keep), class_limit=limit, zero_share_target=zero_share, selected=len(sel),
            digits=dict(sorted(collections.Counter(r["completion"] for r in sel).items())),
            tiers=dict(sorted(collections.Counter(r["tier"] for r in sel).items())),
            drift_share=round(sum(r["is_drift"] for r in sel) / max(1, len(sel)), 3),
            bands=dict(collections.Counter(r["band"] for r in sel)),
            sessions=len({r["session_id"] for r in sel}))

    json.dump(rep, open(os.path.join(out, "prep_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
