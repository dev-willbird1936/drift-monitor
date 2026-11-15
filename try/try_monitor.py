"""Try the drift monitor on your own session.

usage:
  try_monitor.py                         score every example in the examples folder (one per .txt file)
  try_monitor.py my_session.txt          score one file
  try_monitor.py my_session.txt --tier 1 pretend the monitor can only see the agent's replies

Session format (plain text):
  GOAL: what the user asked for
  CONSTRAINT: a rule the user set            (zero or more lines)
  UPDATE after 2: a later user message       (zero or more lines)
  --- step 1
  THINK: the agent's reasoning               (optional)
  CALL bash: {"command": "ls"}               (optional, any tool name)
  RESULT: what the tool returned             (optional)
  SAY: what the agent told the user          (optional)
  --- step 2
  ...

Evidence tiers: 1 replies only, 2 replies + tool names, 3 + calls/results + reasoning summary,
4 + start of reasoning, 5 everything (default), 6 live stream.
"""
import glob, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "submission", "space"))
os.environ.setdefault("HF_HUB_CACHE", "E:/Data/Drift/hf/hub")
os.environ.setdefault("DRIFT_ADAPTER", "E:/Data/Drift/train_run_v4b")
os.environ.setdefault("DRIFT_TOKENIZER", "E:/Data/Drift/train_run/tokenizer")
import drift_score as D

args = [a for a in sys.argv[1:] if not a.startswith("--")]
tier = int(sys.argv[sys.argv.index("--tier") + 1]) if "--tier" in sys.argv else 5
args = [a for a in args if a != str(tier)] if "--tier" in sys.argv else args
files = args or sorted(glob.glob(os.path.join(HERE, "examples", "*.txt")))

print("loading the monitor (about 15 seconds)...", flush=True)
m = D.Monitor()
for f in files:
    goal, cons, ups, steps = D.parse_session(open(f, encoding="utf-8").read())
    if not steps:
        print(f"\n{os.path.basename(f)}: no steps found (each step starts with a line like '--- step 1')")
        continue
    t = time.time()
    traj = m.score_session(goal, cons, ups, steps, tier)
    print(f"\n=== {os.path.basename(f)}  (evidence tier {tier}, {time.time() - t:.1f}s)")
    print(f"goal: {goal[:100]}")
    for r in traj:
        bar = "#" * int(r["score"] / 4)
        flag = "  <-- ALERT at 30" if r["score"] >= 30 else ""
        print(f"  step {r['step']:2}  {r['score']:5.1f}  +/-{r['spread']:4.1f}  {bar}{flag}")
input("\npress Enter to close") if len(sys.argv) == 1 else None
