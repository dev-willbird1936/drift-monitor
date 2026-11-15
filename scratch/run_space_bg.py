import subprocess, os
log = open("E:/Data/Drift/loop_pilot/space_ui.log", "wb")
p = subprocess.Popen(["E:/Data/Drift/venv_space/Scripts/python.exe", "-X", "utf8", "scratch/test_space_ui.py", "submission/space", "http://127.0.0.1:8777", "--keep"],
                     stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                     creationflags=0x08000000 | 0x00000200)
open("E:/Data/Drift/loop_pilot/space_ui.pid", "w").write(str(p.pid)); print(p.pid)
