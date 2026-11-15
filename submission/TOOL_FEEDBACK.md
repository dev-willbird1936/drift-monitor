# Feedback on the Lexsi stack

Prepared for the organisers, who asked for "real-world feedback and feature requests from builders to improve
it further". All of it comes from actually running the three Track 1 libraries end to end on a real project:
a 90k-row dataset, two QLoRA training runs and roughly a dozen evaluation passes.

**Nothing here has been filed.** These are drafts for the user to post as GitHub issues or a Discord ticket.

---

## 1. AlignTune — assistant-only loss masking cannot read Tiny Aya's chat template

**Severity: blocking, with a non-obvious failure mode.**

`CohereLabs/tiny-aya-global` ships `chat_template` in `tokenizer_config.json` as a **dict of named templates**
rather than a single string. TRL's assistant-only loss masking expects a string containing `{% generation %}`
markers, so with `assistant_only_loss=True` the run does not fail loudly — it silently fails to locate the
assistant span.

This is worth fixing because it is the exact combination the hackathon pushes people toward: a Cohere Labs
model, AlignTune's TRL backend, and SFT.

**Workaround we used.** Supply an equivalent single-string template with explicit generation markers, and pass
it via `tokenizer_name_or_path` pointing at a tokenizer directory containing a `chat_template.jinja`:

```
<BOS_TOKEN>{% for m in messages %}...{% if m['role'] == 'assistant' %}...{% generation %}{{ m['content'] }}{% endgeneration %}...
```

**Suggested fix.** When `chat_template` resolves to a dict, either select the default entry and convert it, or
raise a clear error naming the model and the required marker. A silent mis-mask costs a full training run to
discover.

**Second-order trap.** Evaluation must use the same tokenizer. If it does not, the digit token ids shift and
the model appears badly calibrated rather than mis-evaluated. A warning when the eval tokenizer differs from
the training tokenizer would save people a confusing afternoon.

---

## 2. AuditKIT — `NumericAccuracy` is not what the name suggests

**Severity: minor, documentation.**

Building a regression-style scorer, `ak.NumericAccuracy` is the obvious candidate by name. It is actually a
RAG-oriented numeric-support check ("do the answer claims carry every number in the gold answer", RST-06),
which silently reports `not_tested` without a gold answer in the expected shape.

**Suggestion.** Either rename toward its purpose (`NumericSupport`, `NumericClaimCoverage`) or add a one-line
"this is not a regression metric; for that use a custom scorer" note to the docstring.

**What worked well.** Bare callables as scorers is an excellent API. `fn(sample, output) -> float` wrapped
automatically into a `Metric` meant six project-specific metrics took about twenty lines. `PrecomputedModel`
is the single most useful thing in the library for our workflow: we scored every historical run offline from
stored prediction files with no GPU and no re-inference.

---

## 3. AuditKIT — `CustomPredicate` is not accepted by `evaluate()`

**Severity: minor, confusing error.**

`ak.CustomPredicate` is exported at top level, which strongly implies it is a scorer. Passing it to
`ak.evaluate(..., scorers=[...])` raises `TypeError: cannot interpret scorers=<...agent_eval.outcome.CustomPredicate>`.
It belongs to the agent-eval path.

**Suggestion.** Either accept it in `_to_metrics` by adapting it, or raise an error that says which API it
belongs to. The current message does not hint at the fix.

---

## 4. AuditKIT — the `warn` grade earned its keep

Not a complaint. Our before/after had one metric regress: `missed_drift` went 0.000 → 0.033, and
`grade_delta` flagged it `warn` while everything else passed.

That is correct and valuable. Our untrained baseline never misses drift **because it alerts on 84% of clean
steps** — it is degenerate, not good. A single headline accuracy number would have hidden that, and the
`warn` forced us to explain it in the write-up. Direction-aware metric contracts are the right design.

---

## 5. Cross-cutting — provenance chaining is underdocumented

`lexsi_provenance.json` is mentioned in CuratorKIT's README as the mechanism tying stages together, but we
could not find a worked example of what a three-stage chain should contain, or a validator that checks a
chain is intact. We ended up writing our own.

**Suggestion.** A short "chaining three repos" example, plus `ak.read_provenance` gaining a `validate=True`
mode that checks each declared upstream stage actually produced the referenced artifacts. For a hackathon
whose central rule is "chain at least 2 repos into a real pipeline", this is the one piece of documentation
most likely to be wanted.

---

## 6. Smaller notes

- **CuratorKIT**: `pip install curatorkit` core-only worked cleanly; the `[all]` extra pulls a large
  dependency set that conflicted with a pinned `torch` in our training environment. Installing core-only into
  the training env and `[all]` into a separate env was the practical route. Worth a note in the docs.
- **AuditKIT**: zero required dependencies is a genuinely good decision and made it trivial to add late in a
  project. It installed into a pinned torch environment without touching anything.
- **Licensing**: LSAL v1.2 is clear about academic use, but the boundary for "use by any organization" is
  ambiguous for a solo entrant who may later reuse the pipeline commercially. A short FAQ entry would help.
