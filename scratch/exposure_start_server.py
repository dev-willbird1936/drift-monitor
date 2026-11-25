"""Start the drift scoring server hidden (no window, no focus) and wait for /health.

usage: exposure_start_server.py   (writes $DRIFT_DATA/loop_pilot/run_exposure/score_server.pid and .log)
"""
import os, subprocess, sys, time, urllib.request
DATA = os.environ.get("DRIFT_DATA", "data")  # run outputs; set DRIFT_DATA to move them

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
OUT = (DATA + "/loop_pilot/run_exposure")
PORT = os.environ.get("DRIFT_PORT", "8765")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "score_server.log"), "ab")
p = subprocess.Popen([sys.executable, "-X", "utf8", "score_server.py", "--port", PORT],
                     cwd=HARNESS,
                     env=dict(os.environ, DRIFT_ADAPTER=os.environ.get("DRIFT_ADAPTER", "dev-willbird1936/drift-monitor-tiny-aya"),
                              DRIFT_TOKENIZER=os.environ.get("DRIFT_TOKENIZER", "dev-willbird1936/drift-monitor-tiny-aya"), PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=0x08000000 | 0x00000200)
open(os.path.join(OUT, "score_server.pid"), "w").write(str(p.pid))
print("pid", p.pid, flush=True)
t0 = time.time()
while time.time() - t0 < 180:
    if p.poll() is not None:
        print("server exited with code", p.returncode)
        sys.exit(1)
    try:
        print("health", urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read().decode(), f"after {time.time() - t0:.0f}s")
        sys.exit(0)
    except Exception:
        time.sleep(1)
print("no health answer within 180 s")
sys.exit(1)
