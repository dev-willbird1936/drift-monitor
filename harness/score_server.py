"""Local scoring server: holds the drift monitor in memory and scores prompts over HTTP.

usage: score_server.py [--port 8765]
POST /score {"prompt": "..."} -> {"score": 0-100, "spread": float, "ms": int}
POST /score {"session": {goal, constraints, updates, steps, tier}} -> the same; the prompt is built here with render.py
POST /advisor_prompt {"session": {...}, "score": float} -> {"system": str, "prompt": str} for an advisor call
GET  /health -> {"ok": true}

session.steps use render.py's shape: {thinking, say, calls: [{tool, args}], results: [{text, is_error}]}.
updates: [{after_step, text}] for user messages after the first.

Binds to 127.0.0.1 only. Model location comes from DRIFT_BASE / DRIFT_ADAPTER / DRIFT_TOKENIZER
(see submission/space/drift_score.py).
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission", "space"))
import drift_score as D
import driftwatch as W

port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8765
monitor = D.Monitor()
lock = threading.Lock()  # one GPU, one forward pass at a time


class H(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"ok": True})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path.rstrip("/") == "/advisor_prompt":
                w = W.DriftWatch.from_session(req["session"])
                return self._send({"system": W.ADVISOR_SYSTEM, "prompt": w.advisor_prompt(float(req.get("score", 0))),
                                   "note_prefix": W.NOTE_PREFIX})
            prompt = req["prompt"] if "prompt" in req else W.DriftWatch.from_session(req["session"]).prompt()[0]
            t = time.time()
            with lock:
                score, spread, _ = monitor.score(prompt)
            self._send({"score": round(score, 2), "spread": round(spread, 2), "ms": int((time.time() - t) * 1000)})
        except Exception as e:
            self._send({"error": str(e)[:300]}, 500)

    def log_message(self, *a):
        pass


print(f"drift monitor listening on 127.0.0.1:{port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
