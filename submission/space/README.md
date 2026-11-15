---
title: Drift Monitor
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
datasets:
- dev-willbird1936/drift-monitor-data
short_description: Watches an AI agent and calls an advisor when it drifts
---

# Drift monitor

A 3.35B Cohere model (`CohereLabs/tiny-aya-global`, 4-bit QLoRA) that reads an AI agent's session and emits
**one digit, 0–9**: how far the work has drifted from what the user asked for. A threshold on that number
decides when it is worth spending a call on a stronger model.

Built for the Lexsi AI Safety and Alignment Hackathon 2026, Track 1 (Fine-tuning and Data Curation),
problem statement 2, replacing a closed-model API. Pipeline: CuratorKIT → AlignTune → AuditKIT.

## What you can do here

- **Live agent.** A real agent works on a small task in a sandboxed folder (list, read, write, edit and delete
  files; no shell, no network). Pick a hidden drift to push it off course. That instruction goes into the agent's
  system prompt and is never shown to the monitor or the advisor. The monitor scores every step. When the score
  crosses the threshold, an advisor model writes a note and the agent gets it before its next step. Turn the
  advisor off to see what the agent does unwatched. Judges: use the access code from the submission. Anyone can
  paste their own OpenRouter key instead (used for that run only, never stored).
- **Replay.** Five recorded sessions from the closed-loop experiments, each run watched and unwatched with the
  same task and hidden prompt. One of them is a known limit: the agent deletes files before the advice arrives.
- **Score a session.** Paste or write a session and change the evidence tier. A monitor rarely sees everything
  an agent did, and the model is told how much it can see.
- **Use it in your agent.** `driftwatch.py` and `render.py` attach the monitor and an advisor to any agent loop,
  scoring through this Space's `score_prompt` API.

## Results

Measured on real held-out agent sessions (770 sessions never trained on), leak-free:

| | Untrained base | Tuned |
|---|---|---|
| Mean absolute error | 43.9 | **7.8** |
| AUROC | 0.552 | **0.957** |
| Correct alert decisions | 0.164 | **0.922** |
| False alarms | 83.6% | **5.8%** |

Against two frontier models given the identical monitoring prompt on 497 identical rows: at matched recall
the 3.35B model raises **5–6× fewer false alarms**; at matched false-alarm rate it catches **1.6–1.8× more
drift**, at roughly 20× the speed. Giving the frontier models the full scoring rubric did not close the gap.

309 ms per score interactively, 2.66 GB of VRAM.

## Honesty

One example in the dropdown is a **known failure**, labelled as such: a constraint breach where the score
fires and then decays below the alert line while the breach is still unresolved. It is caught at threshold
30 and missed at 40, which is why 30 is the recommended default. Other documented weaknesses: severity is
compressed at the top of the scale, and thin evidence tiers produce over-suspicion.

## Setup

Variables: `DRIFT_ADAPTER` (adapter repo id), `DRIFT_TOKENIZER` if the tokenizer lives elsewhere. Secrets:
`HF_TOKEN` (the base model is gated), and optionally `OPENROUTER_API_KEY` with `DRIFT_ACCESS_CODE` to let
code holders run the live agent on the owner's key, capped by `DRIFT_DAILY_CAP_USD` (default 2.00 per day).

## Licence

The base model is CC-BY-NC; this is non-commercial research. CuratorKIT and AuditKIT are used under the
Lexsi Source Available Licence v1.2.
