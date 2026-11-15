"""Scoring backend for the drift monitor demo page: two API endpoints, no interface to speak of.

  score(session_json)                 -> {"score": 0-100, "spread": float}
  advisor_prompt(session_json, score) -> {"system": str, "prompt": str, "note_prefix": str}

session_json: {"goal", "constraints", "updates": [{after_step, text}], "steps": [{thinking, say, calls: [{tool, args}],
results: [{text, is_error}]}], "tier": 5 or 6, "live": bool}. With tier 6 and live, the latest step's reasoning is the
text streamed so far and is used as is. Prompts are built with render.py, the exact renderer used for the training data.

Space variable: DRIFT_ADAPTER (the adapter repo id). Secret: HF_TOKEN (the base model is gated).
"""
import json
import threading

import gradio as gr

import drift_score as D
import driftwatch as W
import render as R

try:  # ZeroGPU: a GPU is attached only while a spaces.GPU function runs
    import spaces
    gpu = spaces.GPU(duration=10)
except ImportError:
    def gpu(fn):
        return fn

_monitor = {"m": None}
_render_lock = threading.Lock()
MAX_JSON = 200_000


def monitor():
    if _monitor["m"] is None:
        _monitor["m"] = D.Monitor()
    return _monitor["m"]


@gpu
def _score(prompt):
    s, sp, _ = monitor().score(prompt)
    return round(float(s), 1), round(float(sp), 1)


def _watch(session_json):
    if not isinstance(session_json, str) or len(session_json) > MAX_JSON:
        raise gr.Error("session_json must be a JSON string under 200 kB")
    session = json.loads(session_json)
    tier = int(session.get("tier") or 5)
    if tier not in range(1, 7):
        raise gr.Error("tier must be 1-6")
    session["tier"] = tier
    return W.DriftWatch.from_session(session), bool(session.get("live"))


def _prompt(w, live):
    if not (live and w.tier == 6):
        return w.prompt()[0]
    with _render_lock:  # keep the streamed-so-far reasoning exactly as received
        cut = R._stream_cut
        R._stream_cut = lambda t: " ".join(str(t).split())
        try:
            return w.prompt()[0]
        finally:
            R._stream_cut = cut


def score(session_json: str) -> dict:
    """Score the session after its latest step."""
    w, live = _watch(session_json)
    if not w.steps:
        return {"score": 0.0, "spread": 0.0}
    s, sp = _score(_prompt(w, live))
    return {"score": s, "spread": sp}


def advisor_prompt(session_json: str, score: float) -> dict:
    """The advisor prompt for this session, to send to an advisor model of your choice."""
    w, _ = _watch(session_json)
    return {"system": W.ADVISOR_SYSTEM, "prompt": w.advisor_prompt(float(score)), "note_prefix": W.NOTE_PREFIX}


with gr.Blocks(title="Drift monitor scorer") as demo:
    gr.Markdown("# Drift monitor scorer\nBackend for the [drift monitor demo](https://huggingface.co/spaces/dev-willbird1936/drift-monitor). "
                "API endpoints: `score(session_json)` and `advisor_prompt(session_json, score)`. See **Use via API** below.")
    gr.api(score, api_name="score")
    gr.api(advisor_prompt, api_name="advisor_prompt")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
