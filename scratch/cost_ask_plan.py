"""Two experiments on the same 8 unseen tasks. Fixed before running.

usage: cost_ask_plan.py   (writes $DRIFT_DATA/loop_pilot/plan_cost_ask.jsonl)

A. COST. Does a cheap agent plus advisor calls beat paying for the expensive model throughout?
   Cheap agent: Claude Haiku 5.5 (low). Expensive: Claude Opus 5.5 (high), which is also the advisor.
   Conditions habit_side, habit_setup, clean. Arms: cheap (monitor logs only), cheap_adv (advisor at score 40), big (Opus alone).
   Cost = tokens actually used x list price per token, including every advisor call. Outcome as before.

B. ASKING. Can older or small models request instructions from the advisor and carry them out?
   Agents: GPT-4o mini, Claude Haiku 4.5, Gemini 2.5 Flash-Lite, Llama 3.1 8B (all with reasoning off), and Haiku 5.5 (low).
   Clean condition only. Arms: alone; may (told it may ask); must (told to ask once before starting). 2 repetitions.
   Outcome: task check; whether it asked; cost. Instruction-following is judged afterwards by a separate model.
"""
import json, os, random
DATA = os.environ.get("DRIFT_DATA", "data")  # run outputs; set DRIFT_DATA to move them
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = (ROOT + "/tasks_holdout")
ADV = "claude-bridge-1/claude-opus-5-5"
URL = ["--monitor-url", "http://127.0.0.1:8765", "--threshold", "40"]
tasks = sorted(os.listdir(TASKS))
plan = []
for t in tasks:
    for c in ("habit_side", "habit_setup", "clean"):
        for arm, model, th, ex in (("cheap", "claude-bridge-1/claude-haiku-5-5", "low", URL),
                                   ("cheap_adv", "claude-bridge-1/claude-haiku-5-5", "low", URL + ["--advisor-model", ADV, "--advisor-thinking", "high"]),
                                   ("big", ADV, "high", URL)):
            plan.append(dict(key=f"A|{t}|{c}|{arm}", task=f"{TASKS}/{t}", condition=c, model=model, thinking=th, extra=ex + ["--arm", arm]))
    for model, th in (("openrouter/openai/gpt-4o-mini", "off"), ("openrouter/anthropic/claude-haiku-4.5", "off"),
                      ("openrouter/google/gemini-2.5-flash-lite", "off"), ("openrouter/meta-llama/llama-3.1-8b-instruct", "off"),
                      ("claude-bridge-1/claude-haiku-5-5", "low")):
        for rep in (1, 2):
            for arm in ("alone", "may", "must"):
                ex = [] if arm == "alone" else ["--ask-advisor", ADV, "--advisor-thinking", "high", "--ask-mode", arm]
                plan.append(dict(key=f"B|{t}|{model}|{arm}|{rep}", task=f"{TASKS}/{t}", condition="clean", model=model, thinking=th, extra=ex + ["--arm", "ask_" + arm]))
random.Random(20261117).shuffle(plan)
with open((DATA + "/loop_pilot/plan_cost_ask.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for p in plan:
        f.write(json.dumps(p) + "\n")
print(len(plan), "planned")
