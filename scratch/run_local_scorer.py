"""Run the scorer Space app locally for testing the demo page, forwarding scoring to harness/score_server.py.

usage: run_local_scorer.py SCORE_SERVER_URL [PORT]
The gradio environment has no torch, so the model imports are stubbed; prompts are built by the real app code and
scored by the real model behind SCORE_SERVER_URL.
"""
import os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
url, port = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7861
sys.path.insert(0, os.path.join(ROOT, "submission", "space_scorer"))
sys.path.insert(1, os.path.join(ROOT, "submission", "space"))
torch = types.ModuleType("torch")
torch.no_grad = lambda: (lambda f: f)
torch.bfloat16 = None
torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
sys.modules["torch"] = torch
peft = types.ModuleType("peft"); peft.PeftModel = object; sys.modules["peft"] = peft
tf = types.ModuleType("transformers"); tf.AutoModelForCausalLM = tf.AutoTokenizer = tf.BitsAndBytesConfig = object
sys.modules["transformers"] = tf

import app
from driftwatch import LocalServerScorer

srv = LocalServerScorer(url)
app._monitor["m"] = types.SimpleNamespace(score=lambda p: (*srv(p), None))
app.demo.queue(default_concurrency_limit=4).launch(server_name="127.0.0.1", server_port=port, show_error=True)
