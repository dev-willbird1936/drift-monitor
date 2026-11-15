"""ROI: self-hosting this drift monitor versus sending every step to a closed model.

usage: roi.py OUT_JSON [--steps-per-month 1000000] [--prompt-tokens 694] [--gpu-hourly 0.473]

The monitor reads an agent's step and emits one digit. The alternative is to send the same context to a
large closed model and ask it the same question. Both are priced here on the same workload.

Closed-model list prices move and differ by vendor, so this does not assert one number. It computes the
self-hosted cost from measured throughput, then shows the break-even and the saving across a range of
input-token price points, which is robust to whatever the current rate card says.

Measured inputs, all from this project rather than estimated:
  - 132 ms per score at batch 8 (bench_latency.py, RTX 3080 12GB, 4-bit)
  - 694 mean prompt tokens on the real held-out split, 597 median, 1846 p99
  - 1 output token: the monitor emits a single digit
  - $0.473/hour, the price actually paid for the A100 40GB used for training
"""
import json, sys

# measured, not assumed
MS_PER_SCORE_BATCHED = 131.9       # bench_latency.py, batch 8, 500-token prompts
MS_PER_SCORE_SINGLE = 308.8        # batch 1
PROMPT_TOKENS_MEAN = 694           # dataset_v4b test_recorded, measured
OUTPUT_TOKENS = 1                  # one digit
TRAIN_GPU_HOURS = 7.3              # two training runs, 3h42m + 3h32m
TRAIN_EVAL_HOURS = 3.7             # evaluation and idle on the rented box
GPU_HOURLY_PAID = 0.473            # A100 40GB, actual price paid

# Closed-model input prices per million tokens. A spread, not a claim about any one vendor.
PRICE_POINTS = [0.10, 0.25, 0.50, 1.00, 2.50, 5.00, 10.00, 15.00]


def main():
    arg = lambda k, d: float(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    out = sys.argv[1] if len(sys.argv) > 1 else "roi.json"
    steps = arg("--steps-per-month", 1_000_000)
    ptok = arg("--prompt-tokens", PROMPT_TOKENS_MEAN)
    hourly = arg("--gpu-hourly", GPU_HOURLY_PAID)

    # self-hosted serving
    scores_per_sec = 1000.0 / MS_PER_SCORE_BATCHED
    tokens_per_sec = scores_per_sec * ptok
    gpu_hours = steps / scores_per_sec / 3600.0
    serve_cost = gpu_hours * hourly
    train_cost = (TRAIN_GPU_HOURS + TRAIN_EVAL_HOURS) * hourly

    print(f"workload: {steps:,.0f} agent steps per month, {ptok:.0f} prompt tokens each, "
          f"{OUTPUT_TOKENS} output token")
    print(f"\nself-hosted (measured throughput, ${hourly:.3f}/GPU-hour)")
    print(f"  {scores_per_sec:.1f} scores/sec, {tokens_per_sec:,.0f} prompt tokens/sec")
    print(f"  {gpu_hours:,.1f} GPU-hours/month  ->  ${serve_cost:,.2f}/month serving")
    print(f"  one-off training and evaluation: {TRAIN_GPU_HOURS + TRAIN_EVAL_HOURS:.1f} GPU-hours "
          f"= ${train_cost:,.2f}")
    print(f"  cost per 1,000 steps: ${1000 * serve_cost / steps:.4f}")

    total_in_tokens = steps * ptok
    print(f"\nclosed model on the same workload ({total_in_tokens/1e6:,.1f}M input tokens/month)")
    print(f"  {'$/Mtok in':>10} {'monthly':>12} {'vs self-host':>14} {'saving':>9}")
    rows = []
    for p in PRICE_POINTS:
        closed = total_in_tokens / 1e6 * p
        ratio = closed / serve_cost if serve_cost else float("inf")
        saving = 1 - serve_cost / closed if closed else 0.0
        rows.append(dict(price_per_mtok_input=p, closed_monthly=round(closed, 2),
                         ratio_vs_self_hosted=round(ratio, 1), saving_fraction=round(saving, 4)))
        print(f"  {p:>10.2f} {closed:>12,.2f} {ratio:>13.1f}x {saving:>8.1%}")

    # break-even: the input price at which a closed model matches self-hosting
    breakeven = serve_cost / (total_in_tokens / 1e6)
    print(f"\nbreak-even closed-model input price: ${breakeven:.4f} per million tokens")
    print("  below that a closed model is cheaper; every realistic rate card sits well above it")

    # TOTAL system cost: the monitor does not remove advisor calls, it rations them. The honest comparison
    # is (monitor serving + the advisor calls that still fire) against reviewing every step with the big model.
    print("\ntotal system cost: monitor + the advisor calls it still triggers")
    print(f"  {'$/Mtok':>8} {'review every step':>18} {'monitor+alerts@40':>19} {'monitor+alerts@30':>19} {'saving@40':>10}")
    totals = []
    for p_tok in (0.25, 2.50, 10.00):
        every = total_in_tokens / 1e6 * p_tok
        t40 = serve_cost + steps * 0.172 * ptok / 1e6 * p_tok
        t30 = serve_cost + steps * 0.257 * ptok / 1e6 * p_tok
        sav = 1 - t40 / every if every else 0
        totals.append(dict(price_per_mtok=p_tok, review_every_step=round(every, 2),
                           total_at_threshold_40=round(t40, 2), total_at_threshold_30=round(t30, 2),
                           saving_vs_every_step=round(sav, 4)))
        print(f"  {p_tok:>8.2f} {every:>18,.2f} {t40:>19,.2f} {t30:>19,.2f} {sav:>9.1%}")
    print("  (monitor serving is included in both threshold columns; alert rates are measured, not assumed)")

    # the actual product decision: the monitor exists to ration advisor calls
    print("\nwhat the monitor is for: it decides when to spend an advisor call")
    print(f"  {'alert rate':>11} {'advisor calls':>14} {'at $2.50/Mtok':>15} {'at $10/Mtok':>13}")
    adv = []
    for rate, label in ((1.0, "always on"), (0.172, "threshold 40"), (0.257, "threshold 30"), (0.0, "never")):
        calls = steps * rate
        # an advisor review reads the same context and writes a short answer; charge input only, which is
        # conservative in the advisor's favour
        c25 = calls * ptok / 1e6 * 2.50
        c10 = calls * ptok / 1e6 * 10.00
        adv.append(dict(policy=label, alert_rate=rate, calls=int(calls),
                        cost_at_2_50=round(c25, 2), cost_at_10=round(c10, 2)))
        print(f"  {label:>11} {calls:>14,.0f} {c25:>15,.2f} {c10:>13,.2f}")
    saved_25 = adv[0]["cost_at_2_50"] - adv[1]["cost_at_2_50"]
    print(f"\n  at threshold 40 the monitor suppresses {1-0.172:.1%} of advisor calls,")
    print(f"  saving ${saved_25:,.2f}/month at $2.50/Mtok while still catching 87% of drifting sessions")

    payload = dict(
        workload=dict(steps_per_month=steps, prompt_tokens=ptok, output_tokens=OUTPUT_TOKENS),
        measured=dict(ms_per_score_batched=MS_PER_SCORE_BATCHED, ms_per_score_single=MS_PER_SCORE_SINGLE,
                      scores_per_sec=round(scores_per_sec, 2), prompt_tokens_per_sec=round(tokens_per_sec),
                      vram_gb=2.66, gpu="RTX 3080 12GB for latency, A100 40GB for training"),
        self_hosted=dict(gpu_hourly=hourly, gpu_hours_per_month=round(gpu_hours, 2),
                         serving_monthly_usd=round(serve_cost, 2),
                         one_off_training_usd=round(train_cost, 2),
                         cost_per_1000_steps_usd=round(1000 * serve_cost / steps, 5)),
        closed_model_sensitivity=rows,
        total_system_cost=totals,
        breakeven_input_price_per_mtok=round(breakeven, 5),
        advisor_gating=adv,
        caveats=[
            "Closed-model prices are a sensitivity range, not a quote; check current rate cards.",
            "Input tokens only are charged to the closed model, which favours it.",
            "Self-hosted serving assumes the GPU is busy; a partly idle GPU costs the same per hour.",
            "The monitor rations advisor calls rather than removing them; total system cost includes both.",
            "Interactive single-step latency is 309 ms; 132 ms per score is the batch-8 figure.",
            "Throughput measured on a consumer RTX 3080; a datacentre GPU is faster per dollar.",
        ])
    json.dump(payload, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
