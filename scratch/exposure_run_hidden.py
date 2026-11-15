"""Start a harness script hidden (no window, no focus), logging to a file; print its pid.

usage: exposure_run_hidden.py LOG PIDFILE SCRIPT [ARGS...]   (SCRIPT is relative to harness/, run with the system Python)
"""
import os, subprocess, sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
log_path, pid_path, args = sys.argv[1], sys.argv[2], sys.argv[3:]
log = open(log_path, "ab")
p = subprocess.Popen([sys.executable, "-X", "utf8"] + args, cwd=HARNESS,
                     env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"),
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=0x08000000 | 0x00000200)
open(pid_path, "w").write(str(p.pid))
print("pid", p.pid)
