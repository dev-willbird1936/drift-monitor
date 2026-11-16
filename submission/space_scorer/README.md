---
title: Drift Monitor Scorer
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
- CohereLabs/tiny-aya-global
- dev-willbird1936/drift-monitor-tiny-aya
short_description: Scoring API behind the drift monitor demo
---

Scoring backend for the [drift monitor demo](https://huggingface.co/spaces/dev-willbird1936/drift-monitor).
It holds the monitor model (on ZeroGPU) and exposes these API endpoints:

- `score(session_json)`: the drift score 0-100 after the session's latest step.
- `advisor_prompt(session_json, score)`: the prompt to send to an advisor model.
- `agent_chat(payload_json)` and `advise(session_json, score, advisor_model)`: the demo's agent and advisor calls, made
  with a shared OpenRouter key so visitors need no key. Only the demo's models are allowed, output is capped, and
  spend is capped per visitor and in total; the key also has its own hard limit on OpenRouter.
- `status()`: whether the shared key is available to this visitor.

Settings: variable `DRIFT_ADAPTER=dev-willbird1936/drift-monitor-tiny-aya`; secrets `HF_TOKEN` (the base model is gated)
and `OPENROUTER_API_KEY` (the shared key). Optional variables: `DRIFT_TOTAL_CAP_USD` (default 18),
`DRIFT_VISITOR_CAP_USD` (default 1.5), `DRIFT_CALLS_PER_HOUR` (default 150).
