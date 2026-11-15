"""Score the drift monitor with AuditKIT, the third Lexsi repo in the Track 1 pipeline.

usage: auditkit_eval.py OUT_DIR [--results-dir E:/Data/Drift/results] [--threshold 40] [--limit N]

CuratorKIT curated the training data, AlignTune trained the adapter, and AuditKIT grades the result here.
It runs entirely offline on prediction files already written by eval_v4.py, using AuditKIT's PrecomputedModel
backend, so no GPU and no network are involved.

Six metrics, chosen because they are what the monitor is actually judged on rather than what is easy to
report:

  exact_digit        the model emits one digit 0-9; did it emit the right one          (maximize)
  within_one_digit   a neighbouring digit is a usable answer for a running score       (maximize)
  abs_error_points   mean absolute error on the underlying 0-100 scale                 (minimize)
  alert_decision     does the advisor call fire when, and only when, it should         (maximize)
  missed_drift       drift present, no alert: the expensive failure                    (minimize)
  false_alarm        no drift, alert raised: the wasteful failure                      (minimize)

Nothing is uploaded or published. The HF publishing helpers in AuditKIT are deliberately not imported.
"""
import json, os, sys

import auditkit as ak
from auditkit.api import FunctionScorer
from auditkit.model import PrecomputedModel

THRESHOLD = 40.0


def digit(score):
    return min(9, max(0, int(float(score) // 10)))


def load_preds(path, limit=0):
    """Turn an eval_v4.py prediction file into AuditKIT samples."""
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw["preds"]
    if limit:
        rows = rows[:limit]
    samples = []
    for r in rows:
        samples.append(ak.Sample(
            input=str(r["id"]),
            target=str(digit(r["drift_score"])),
            id=str(r["id"]),
            actual_output=str(digit(r["pred"])),
            metadata=dict(true_score=float(r["drift_score"]), pred_score=float(r["pred"]),
                          spread=float(r.get("spread") or 0.0), tier=r.get("tier"),
                          mode=r.get("mode"), band=r.get("band")),
            tags=[f"tier{r.get('tier')}", str(r.get("band"))]))
    return samples, raw["summary"]


# ---- metrics -------------------------------------------------------------------------------------
# Each reads the continuous scores from metadata where the 0-100 scale matters, and the emitted digit
# where the model's literal output matters.

def exact_digit(sample, output):
    return float(str(output).strip() == str(sample.target).strip())


def within_one_digit(sample, output):
    try:
        return float(abs(int(str(output).strip()) - int(str(sample.target).strip())) <= 1)
    except ValueError:
        return 0.0


def abs_error_points(sample, output):
    m = sample.metadata
    return abs(m["pred_score"] - m["true_score"])


def alert_decision(sample, output):
    m = sample.metadata
    return float((m["pred_score"] >= THRESHOLD) == (m["true_score"] >= THRESHOLD))


def missed_drift(sample, output):
    m = sample.metadata
    return float(m["true_score"] >= THRESHOLD and m["pred_score"] < THRESHOLD)


def false_alarm(sample, output):
    m = sample.metadata
    return float(m["true_score"] < THRESHOLD and m["pred_score"] >= THRESHOLD)


def scorers():
    mx, mn = ak.Direction.MAXIMIZE, ak.Direction.MINIMIZE
    return [
        FunctionScorer(exact_digit, name="exact_digit", direction=mx),
        FunctionScorer(within_one_digit, name="within_one_digit", direction=mx),
        FunctionScorer(abs_error_points, name="abs_error_points", direction=mn),
        FunctionScorer(alert_decision, name="alert_decision", direction=mx),
        FunctionScorer(missed_drift, name="missed_drift", direction=mn),
        FunctionScorer(false_alarm, name="false_alarm", direction=mn),
    ]


def headline(result):
    out = {}
    for name, stat in result.stats.items():
        out[name] = round(float(getattr(stat, "mean", stat)), 4)
    return out


def main():
    arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
    out_dir = sys.argv[1]
    res_dir = arg("--results-dir", os.environ.get("DRIFT_RESULTS", "./results"))
    limit = int(arg("--limit", "0"))
    global THRESHOLD
    THRESHOLD = float(arg("--threshold", "40"))
    os.makedirs(out_dir, exist_ok=True)

    # model label -> prediction file per split
    runs = {
        "base_untrained": os.path.join(res_dir, "v4run1", "base_{split}.json"),
        "run1_synthetic_only": os.path.join(res_dir, "v4run1", "tuned_{split}.json"),
        "run2_final": os.path.join(res_dir, "v4run2", "run2_{split}.json"),
    }
    splits = ["test", "test_category", "test_recorded", "val"]

    table, results = {}, {}
    for split in splits:
        for label, tmpl in runs.items():
            path = tmpl.format(split=split)
            if not os.path.exists(path):
                print(f"  skip {label}/{split}: no prediction file")
                continue
            samples, _ = load_preds(path, limit)
            r = ak.evaluate(samples, PrecomputedModel(), scorers(),
                            experiment_name=f"drift-{split}", tags=[label, split], cache=False)
            h = headline(r)
            table.setdefault(split, {})[label] = h
            results[(split, label)] = r
            print(f"{split:16} {label:20} n={len(samples):5} "
                  f"exact={h['exact_digit']:.3f} within1={h['within_one_digit']:.3f} "
                  f"mae={h['abs_error_points']:.2f} alert={h['alert_decision']:.3f} "
                  f"miss={h['missed_drift']:.3f} false={h['false_alarm']:.3f}", flush=True)
            if r.unscored:
                print(f"    unscored: {r.unscored}")

    # before and after, graded by AuditKIT's own delta thresholds
    print("\nbefore/after deltas (base_untrained -> run2_final), graded by AuditKIT:")
    deltas = {}
    dirs = {"exact_digit": ak.Direction.MAXIMIZE, "within_one_digit": ak.Direction.MAXIMIZE,
            "alert_decision": ak.Direction.MAXIMIZE, "abs_error_points": ak.Direction.MINIMIZE,
            "missed_drift": ak.Direction.MINIMIZE, "false_alarm": ak.Direction.MINIMIZE}
    for split in splits:
        if split not in table or "base_untrained" not in table[split] or "run2_final" not in table[split]:
            continue
        b, c = table[split]["base_untrained"], table[split]["run2_final"]
        deltas[split] = {}
        print(f"  {split}")
        for m, direction in dirs.items():
            d = c[m] - b[m]
            grade = ak.grade_delta(d if direction is ak.Direction.MAXIMIZE else -d, direction=ak.Direction.MAXIMIZE)
            g = getattr(grade, "value", getattr(grade, "name", str(grade)))
            deltas[split][m] = dict(base=b[m], final=c[m], delta=round(d, 4), grade=str(g))
            print(f"    {m:18} {b[m]:>8.3f} -> {c[m]:>8.3f}   delta {d:+8.3f}   {g}")

    # AuditKIT's own report rendering for the headline run
    key = ("test_recorded", "run2_final")
    if key in results:
        try:
            rep = ak.Report(results[key])
            text = rep.render() if hasattr(rep, "render") else str(rep)
            open(os.path.join(out_dir, "auditkit_report_test_recorded.txt"), "w", encoding="utf-8").write(str(text))
            print("\nwrote AuditKIT report for the real held-out split")
        except Exception as e:
            print(f"\nreport rendering skipped: {str(e)[:120]}")

    payload = dict(threshold=THRESHOLD, auditkit_version=getattr(ak, "__version__", "?"),
                   metrics=table, before_after=deltas,
                   run_fingerprints={f"{s}|{l}": dict(run_id=r.run_id, fingerprint=r.fingerprint)
                                     for (s, l), r in results.items()},
                   note="Scored offline from stored predictions via AuditKIT PrecomputedModel. "
                        "No network, no publishing.")
    json.dump(payload, open(os.path.join(out_dir, "auditkit_metrics.json"), "w"), indent=1)

    # provenance, so the three Lexsi stages chain into one verifiable pipeline
    prov = dict(pipeline=["CuratorKIT", "AlignTune", "AuditKIT"],
                stage="AuditKIT",
                auditkit_version=getattr(ak, "__version__", "?"),
                inputs={f"{s}|{l}": runs[l].format(split=s) for (s, l) in results},
                runs={f"{s}|{l}": r.fingerprint for (s, l), r in results.items()})
    json.dump(prov, open(os.path.join(out_dir, "lexsi_provenance_auditkit.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/auditkit_metrics.json and lexsi_provenance_auditkit.json")


if __name__ == "__main__":
    main()
