"""Stop training if the GPU stays full (the sign of spilling into system memory). usage: watchdog.py PID"""
import os
import subprocess, sys, time
pid, full, LOG = sys.argv[1], 0, os.environ.get("DRIFT_WATCHDOG_LOG", "./watchdog.log")
while True:
    time.sleep(60)
    if str(pid) not in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout:
        break
    used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()[0])
    full = full + 1 if used >= 12100 else 0
    open(LOG, "a").write(time.strftime("%H:%M:%S") + f" {used} MiB full_minutes={full}\n")
    if full >= 4:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"])
        open(LOG, "a").write("KILLED: GPU memory full for 4 minutes\n")
        break
