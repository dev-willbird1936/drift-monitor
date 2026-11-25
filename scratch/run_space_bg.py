import subprocess, os
import sys
DATA = os.environ.get("DRIFT_DATA", "data")  # run outputs; set DRIFT_DATA to move them
log = open((DATA + "/loop_pilot/space_ui.log"), "wb")
p = subprocess.Popen([sys.executable, "-X", "utf8", "scratch/test_space_ui.py", "submission/space", "http://127.0.0.1:8777", "--keep"],
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                     creationflags=0x08000000 | 0x00000200)
open((DATA + "/loop_pilot/space_ui.pid"), "w").write(str(p.pid)); print(p.pid)
