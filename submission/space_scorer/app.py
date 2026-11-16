"""Backend for the drift monitor demo page: the monitor model, plus an OpenRouter proxy so visitors need no key.

  score(session_json)                       -> {"score": 0-100, "spread": float}
  advisor_prompt(session_json, score)       -> {"system": str, "prompt": str, "note_prefix": str}
  agent_chat(payload_json)                  -> streams {done, reasoning, content}; the last item adds tool_calls, cost
  advise(session_json, score, advisor)      -> {"text": str, "cost": float}
  status()                                  -> {"shared_key": bool}

session_json: {"goal", "constraints", "updates": [{after_step, text}], "steps": [{thinking, say, calls: [{tool, args}],
results: [{text, is_error}]}], "tier": 5 or 6, "live": bool}. With tier 6 and live, the latest step's reasoning is the
text streamed so far and is used as is. Prompts are built with render.py, the exact renderer used for the training data.

Space variable: DRIFT_ADAPTER (the adapter repo id). Secrets: HF_TOKEN (the base model is gated) and OPENROUTER_API_KEY
(the demo's shared key, which should have its own spend limit on OpenRouter).
"""
try:  # ZeroGPU: `spaces` must be imported before anything that touches CUDA
    import spaces
except ImportError:
    spaces = None

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Iterator

import gradio as gr

import drift_score as D
import driftwatch as W
import render as R

if spaces is not None:  # ZeroGPU: a GPU is attached only while a spaces.GPU function runs
    gpu = spaces.GPU(duration=5)
else:
    def gpu(fn):
        return fn

_monitor = {"m": None}
_render_lock = threading.Lock()
MAX_JSON = 200_000


def monitor():
    if _monitor["m"] is None:
        _monitor["m"] = D.Monitor()
    return _monitor["m"]


if spaces is not None and os.environ.get("SPACES_ZERO_GPU"):
    # ZeroGPU: load once at start-up and place on CUDA here; the GPU is attached inside each spaces.GPU call.
    _m = monitor()
    _m.model.to("cuda")
    _m.lazy_cuda = False


@gpu
def _score(prompt):
    s, sp, _ = monitor().score(prompt)
    return round(float(s), 1), round(float(sp), 1)


def score_gpu(prompt: str) -> list:
    """Score one rendered prompt on the GPU. Returns [score, spread]."""
    if not isinstance(prompt, str) or len(prompt) > MAX_JSON:
        raise gr.Error("prompt must be a string under 200 kB")
    return list(_score(prompt))


# ZeroGPU charges GPU time to whoever makes the request, and anonymous visitors get very little. On the Space, route
# scoring through this Space's own API with the owner's token, so the owner's quota pays and visitors need no login.
SELF_URL = ("https://" + os.environ["SPACE_HOST"]) if os.environ.get("SPACE_HOST") else ""
OWNER_TOKEN = os.environ.get("HF_TOKEN", "")


def _score_as_owner(prompt):
    hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + OWNER_TOKEN}
    req = urllib.request.Request(f"{SELF_URL}/gradio_api/call/score_gpu", json.dumps({"data": [prompt]}).encode(), hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        eid = json.loads(r.read())["event_id"]
    req = urllib.request.Request(f"{SELF_URL}/gradio_api/call/score_gpu/{eid}", headers={"Authorization": "Bearer " + OWNER_TOKEN})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = r.read().decode("utf-8", "replace")
    ev, data = "", None
    for line in text.splitlines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:") and ev in ("complete", "error"):
            data = line[5:].strip()
            break
    if ev != "complete" or not data:
        try:
            msg = json.loads(data).get("error") if data else None
        except ValueError:
            msg = data
        raise gr.Error("scoring failed: " + str(msg or text[:200]))
    s, sp = json.loads(data)[0]
    return float(s), float(sp)


def _score_any(prompt):
    if spaces is not None and SELF_URL and OWNER_TOKEN:
        return _score_as_owner(prompt)
    return _score(prompt)


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
    s, sp = _score_any(_prompt(w, live))
    return {"score": s, "spread": sp}


def advisor_prompt(session_json: str, score: float) -> dict:
    """The advisor prompt for this session, to send to an advisor model of your choice."""
    w, _ = _watch(session_json)
    return {"system": W.ADVISOR_SYSTEM, "prompt": w.advisor_prompt(float(score)), "note_prefix": W.NOTE_PREFIX}


# ---------------------------------------------------------------------------------------------------------------------
# OpenRouter proxy, so visitors can run the live demo without a key of their own. The key is a Space secret and never
# leaves the server. Only the demo's models are allowed, output is capped, and spend is capped per visitor and in total.
# The key itself also has a hard spend limit on OpenRouter.
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
AGENTS = {"openai/gpt-6-luna": "xhigh", "anthropic/claude-haiku-5.5": "xhigh"}
ADVISORS = {"anthropic/claude-opus-5.5": "high", "openai/gpt-6.1-sol": "high"}
TOTAL_CAP = float(os.environ.get("DRIFT_TOTAL_CAP_USD", "18"))
VISITOR_CAP = float(os.environ.get("DRIFT_VISITOR_CAP_USD", "1.5"))
CALLS_PER_HOUR = int(os.environ.get("DRIFT_CALLS_PER_HOUR", "150"))
_spend = {"total": 0.0, "by_ip": {}, "calls": {}}
_spend_lock = threading.Lock()


def _ip(request):
    if request is None:
        return "?"
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _check(ip):
    if not OR_KEY:
        raise gr.Error("This Space has no shared key configured. Paste your own OpenRouter key.")
    now = time.time()
    with _spend_lock:
        if _spend["total"] >= TOTAL_CAP:
            raise gr.Error("The demo's shared budget is used up. Paste your own OpenRouter key to keep going.")
        calls = [c for c in _spend["calls"].get(ip, []) if now - c < 3600]
        if _spend["by_ip"].get(ip, 0.0) >= VISITOR_CAP or len(calls) >= CALLS_PER_HOUR:
            raise gr.Error("You have reached this demo's limit on the shared key. Paste your own OpenRouter key to keep going.")
        calls.append(now)
        _spend["calls"][ip] = calls


def _add_cost(ip, cost):
    with _spend_lock:
        _spend["total"] += cost
        _spend["by_ip"][ip] = _spend["by_ip"].get(ip, 0.0) + cost


def _openrouter(body):
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(), {
        "Authorization": "Bearer " + OR_KEY, "Content-Type": "application/json",
        "HTTP-Referer": "https://huggingface.co/spaces/dev-willbird1936/drift-monitor", "X-Title": "drift-monitor demo"})
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise gr.Error(f"OpenRouter {e.code}: {e.read()[:200].decode('utf-8', 'replace')}")


def _messages(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 80:
        raise gr.Error("messages must be a list of 1 to 80 items")
    if len(json.dumps(messages)) > 300_000:
        raise gr.Error("conversation too long")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            raise gr.Error("bad message")
    return messages


def agent_chat(payload_json: str, request: gr.Request) -> Iterator[dict]:
    """Stream one agent turn through the demo's key: yields {done, reasoning, content}, then the whole message."""
    p = json.loads(payload_json)
    model = p.get("model")
    if model not in AGENTS:
        raise gr.Error("model not offered in this demo")
    tools = p.get("tools") or []
    if len(json.dumps(tools)) > 8000:
        raise gr.Error("tools too large")
    ip = _ip(request)
    _check(ip)
    body = {"model": model, "messages": _messages(p.get("messages")), "tools": tools, "stream": True, "max_tokens": 16000,
            "usage": {"include": True}, "reasoning": {"effort": AGENTS[model]}}
    content, reasoning, cost, calls, details, last = "", "", 0.0, [], [], 0.0
    try:
        with _openrouter(body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if j.get("error"):
                    err = j["error"]
                    raise gr.Error("OpenRouter: " + str(err.get("message", err) if isinstance(err, dict) else err)[:200])
                if (j.get("usage") or {}).get("cost"):
                    cost = float(j["usage"]["cost"])
                d = ((j.get("choices") or [{}])[0].get("delta")) or {}
                reasoning += d.get("reasoning") or ""
                content += d.get("content") or ""
                for rd in d.get("reasoning_details") or []:
                    k = rd.get("index", len(details))
                    ex = next((x for x in details if x.get("index", -1) == k and x.get("type") == rd.get("type")), None)
                    if ex is None:
                        details.append(dict(rd))
                        continue
                    for f in ("text", "summary", "data"):
                        if rd.get(f):
                            ex[f] = (ex.get(f) or "") + rd[f]
                    if rd.get("signature"):
                        ex["signature"] = rd["signature"]
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    while len(calls) <= i:
                        calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    c, fn = calls[i], tc.get("function") or {}
                    c["id"] = tc.get("id") or c["id"]
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += fn.get("arguments") or ""
                now = time.time()
                if now - last > 0.25 and (d.get("reasoning") or d.get("content")):
                    last = now
                    yield {"done": False, "reasoning": reasoning, "content": content}
    finally:
        _add_cost(ip, cost)
    yield {"done": True, "reasoning": reasoning, "content": content, "tool_calls": calls, "reasoning_details": details,
           "cost": cost}


def advise(session_json: str, score: float, advisor_model: str, request: gr.Request) -> dict:
    """Ask the advisor about this session through the demo's key. Returns {text, cost}."""
    if advisor_model not in ADVISORS:
        raise gr.Error("advisor not offered in this demo")
    ip = _ip(request)
    _check(ip)
    w, _ = _watch(session_json)
    body = {"model": advisor_model, "max_tokens": 8000, "usage": {"include": True},
            "reasoning": {"effort": ADVISORS[advisor_model]},
            "messages": [{"role": "system", "content": W.ADVISOR_SYSTEM},
                         {"role": "user", "content": w.advisor_prompt(float(score))}]}
    with _openrouter(body) as r:
        j = json.loads(r.read())
    cost = float((j.get("usage") or {}).get("cost") or 0)
    _add_cost(ip, cost)
    return {"text": (j["choices"][0]["message"].get("content") or "").strip(), "cost": cost}


def status(request: gr.Request) -> dict:
    """Whether the demo's shared key can be used right now by this visitor."""
    ip = _ip(request)
    with _spend_lock:
        ok = bool(OR_KEY) and _spend["total"] < TOTAL_CAP and _spend["by_ip"].get(ip, 0.0) < VISITOR_CAP
    return {"shared_key": ok}


with gr.Blocks(title="Drift monitor scorer") as demo:
    gr.Markdown("# Drift monitor scorer\nBackend for the [drift monitor demo](https://huggingface.co/spaces/dev-willbird1936/drift-monitor). "
                "API endpoints: `score`, `advisor_prompt`, `agent_chat`, `advise` and `status`. See **Use via API** below.")
    gr.api(score, api_name="score", concurrency_limit=16)
    gr.api(score_gpu, api_name="score_gpu")
    gr.api(advisor_prompt, api_name="advisor_prompt")
    gr.api(agent_chat, api_name="agent_chat", concurrency_limit=24)
    gr.api(advise, api_name="advise", concurrency_limit=12)
    gr.api(status, api_name="status", concurrency_limit=24)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
