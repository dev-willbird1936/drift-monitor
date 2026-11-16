"""Scoring core for the drift monitor demo. No UI dependencies, so it can be tested on its own.

The model emits one token: a digit 0-9. We read the softmax over just those ten token ids and take a
probability-weighted mean of the band centres (10*d + 5), which gives a continuous 0-100 score. The spread of
that distribution is returned alongside as an uncertainty signal.
"""
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.environ.get("DRIFT_BASE", "CohereLabs/tiny-aya-global")
ADAPTER = os.environ.get("DRIFT_ADAPTER", "")          # set to your published adapter repo id
TOKENIZER = os.environ.get("DRIFT_TOKENIZER", "")      # defaults to the adapter repo

import render as R  # the exact renderer used to build the training data


def to_render_steps(steps):
    """Steps from parse_session (calls as 'tool: {json}' strings) -> the dict shape render.py expects."""
    out = []
    for s in steps:
        calls = []
        for c in s.get("calls") or []:
            tool, _, raw = str(c).partition(":")
            try:
                args = json.loads(raw) if raw.strip() else {}
            except ValueError:
                args = {"raw": raw.strip()}
            calls.append({"tool": tool.strip(), "args": args})
        out.append(dict(n=s["n"], thinking=s.get("think") or "", thinking_summary=s.get("think") or "", say=s.get("say") or "",
                        calls=calls, results=[{"text": r} for r in s.get("results") or []]))
    return out


def build_prompt(goal, constraints, updates, steps, tier, latest):
    """steps: list of dicts with n, think, say, calls (list of str), results (list of str)."""
    rs = to_render_steps(steps)
    kept, omitted = R.budget_steps(rs, latest)
    return R.build_prompt(goal, constraints, updates, kept, tier, latest, omitted)


class Monitor:
    def __init__(self, base=BASE, adapter=ADAPTER, tokenizer=None, four_bit=True):
        tok_src = tokenizer or TOKENIZER or adapter or base
        self.tok = AutoTokenizer.from_pretrained(tok_src)
        kw = {}
        # ZeroGPU: no GPU exists at start-up, so load in bf16 on CPU and move to the GPU inside the first
        # scoring call. Everywhere else: 4-bit on the local GPU when there is one.
        self.lazy_cuda = bool(os.environ.get("SPACES_ZERO_GPU"))
        if self.lazy_cuda:
            kw["torch_dtype"] = torch.bfloat16
        elif four_bit and torch.cuda.is_available():
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            kw["device_map"] = {"": 0}
        else:
            kw["torch_dtype"] = torch.bfloat16  # CPU: half the memory of float32 (fits a 16 GB free Space)
        model = AutoModelForCausalLM.from_pretrained(base, **kw)
        if adapter:
            # ZeroGPU reports CUDA as available outside GPU calls, so keep the adapter on the CPU until the move
            model = PeftModel.from_pretrained(model, adapter, **({"torch_device": "cpu"} if self.lazy_cuda else {}))
        self.model = model.eval()
        self.digits = [self.tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        self.centres = np.arange(10) * 10 + 5

    @torch.no_grad()
    def score(self, prompt):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        if self.lazy_cuda and torch.cuda.is_available():
            self.model.to("cuda")
            self.lazy_cuda = False
        if torch.cuda.is_available():
            ids = ids.cuda()
        logits = self.model(ids, use_cache=False).logits[0, -1, self.digits].float()
        p = torch.softmax(logits, -1).cpu().numpy()
        mean = float((p * self.centres).sum())
        spread = float(np.sqrt((p * (self.centres - mean) ** 2).sum()))
        del ids, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # long sessions otherwise fragment a small card and spill into system RAM
        return mean, spread, p

    def score_session(self, goal, constraints, updates, steps, tier):
        """Score after every step, so the caller can show the trajectory."""
        out = []
        for k in range(1, len(steps) + 1):
            ups = [u for u in updates if u["after_step"] < k]
            prompt = build_prompt(goal, constraints, ups, steps[:k], tier, k)
            mean, spread, _ = self.score(prompt)
            out.append(dict(step=k, score=round(mean, 1), spread=round(spread, 1)))
        return out


def parse_session(text):
    """Parse the demo's plain-text session format into steps.

    GOAL: ...
    CONSTRAINT: ...            (zero or more)
    UPDATE after 2: ...        (zero or more)
    --- step 1
    THINK: ...
    CALL read: {"path": "x"}
    RESULT: ...
    SAY: ...
    """
    goal, constraints, updates, steps = "", [], [], []
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("goal:"):
            goal = line.split(":", 1)[1].strip()
        elif low.startswith("constraint:"):
            constraints.append(line.split(":", 1)[1].strip())
        elif low.startswith("update after"):
            head, text_part = line.split(":", 1)
            n = "".join(ch for ch in head if ch.isdigit()) or "0"
            updates.append(dict(after_step=int(n), text=text_part.strip()))
        elif low.startswith("---"):
            n = "".join(ch for ch in line if ch.isdigit())
            cur = dict(n=int(n) if n else len(steps) + 1, think="", say="", calls=[], results=[])
            steps.append(cur)
        elif cur is not None:
            if low.startswith("think:"):
                cur["think"] = line.split(":", 1)[1].strip()
            elif low.startswith("say:"):
                cur["say"] = line.split(":", 1)[1].strip()
            elif low.startswith("call "):
                cur["calls"].append(line[5:].strip())
            elif low.startswith("result:"):
                cur["results"].append(line.split(":", 1)[1].strip())
    for i, s in enumerate(steps, 1):
        s["n"] = i
    return goal, constraints, updates, steps


def load_examples(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples.json")
    return json.load(open(path, encoding="utf-8"))
