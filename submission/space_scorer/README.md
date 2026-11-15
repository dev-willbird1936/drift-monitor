---
title: Drift Monitor Scorer
emoji: 📉
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
- CohereLabs/tiny-aya-global
- dev-willbird1936/drift-monitor-tiny-aya
short_description: Scoring API behind the drift monitor demo
---

Scoring backend for the [drift monitor demo](https://huggingface.co/spaces/dev-willbird1936/drift-monitor).
It holds the monitor model and exposes two API endpoints: `score(session_json)` returns the drift score 0-100 after
the session's latest step, and `advisor_prompt(session_json, score)` returns the prompt to send to an advisor model.

Settings: variable `DRIFT_ADAPTER=dev-willbird1936/drift-monitor-tiny-aya`; secret `HF_TOKEN` (the base model is gated).
