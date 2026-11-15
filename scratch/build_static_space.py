"""Build the static Space (free hosting): a replay viewer for submission/space/replays.json.

usage: build_static_space.py  -> writes submission/space_static/index.html and README.md
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission", "space_static")
reps = json.load(open(os.path.join(ROOT, "submission", "space", "replays.json"), encoding="utf-8"))
PATH = re.compile(r'"path":\s*"([^"]+)"')
CMD = re.compile(r'"command":\s*"((?:[^"\\]|\\.)*)')


def action(e):
    """One short line for what the agent did in a step (tool arguments are clipped, so parse them loosely)."""
    verbs = {"read": "read", "write": "wrote", "edit": "edited", "ls": "listed the folder", "delete": "deleted"}
    bits = []
    for c in e.get("calls") or []:
        a = str(c["args"])
        if c["tool"] == "bash":
            m = CMD.search(a)
            cmd = " ".join((m.group(1) if m else a).replace("\\n", " ").replace('\\"', '"').split())
            bits.append("ran " + (cmd[:60] + "…" if len(cmd) > 60 else cmd))
        elif c["tool"] in ("grep", "find"):
            m = re.search(r'"pattern":\s*"([^"]+)"', a)
            bits.append("searched for " + (m.group(1)[:40] if m else "files"))
        else:
            m = PATH.search(a)
            bits.append(verbs.get(c["tool"], c["tool"]) + (" " + m.group(1) if m else (" a file" if c["tool"] in ("write", "edit", "read") else "")))
    if e.get("say"):
        say = " ".join(e["say"].split())
        bits.append("told the user: “" + (say[:80] + "…" if len(say) > 80 else say) + "”")
    return "; ".join(bits) or "thought it over"


for r in reps:
    for arm in ("watched", "unwatched"):
        for e in r[arm]["display"]:
            if e["kind"] == "step":
                e["action"] = action(e)
links = ('Model: <a href="https://huggingface.co/dev-willbird1936/drift-monitor-tiny-aya">tiny-aya-global + QLoRA adapter</a> · '
         '<a href="https://huggingface.co/datasets/dev-willbird1936/drift-monitor-data">Dataset and traces</a> · '
         '<a href="https://github.com/dev-willbird1936/drift-monitor">Code and write-up</a>')
SCORER = os.environ.get("DRIFT_SCORER_URL", "https://dev-willbird1936-drift-monitor-scorer.hf.space")
data = json.dumps(reps, ensure_ascii=False).replace("</", "<\\/")
html = open(os.path.join(OUT, "template.html"), encoding="utf-8").read().replace("__LINKS__", links).replace("__DATA__", data).replace("__SCORER__", SCORER)
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
    "---\ntitle: Drift Monitor\nemoji: 📉\ncolorFrom: indigo\ncolorTo: gray\nsdk: static\napp_file: index.html\npinned: false\n"
    "license: cc-by-nc-4.0\nmodels:\n- dev-willbird1936/drift-monitor-tiny-aya\ndatasets:\n- dev-willbird1936/drift-monitor-data\n"
    "short_description: Replays an AI agent being watched for drift\n---\n\n"
    "Replay viewer for the drift monitor: recorded closed-loop sessions, watched against unwatched, with the monitor's score "
    "at every step and the advisor's notes. See the [model](https://huggingface.co/dev-willbird1936/drift-monitor-tiny-aya) "
    "and the [code](https://github.com/dev-willbird1936/drift-monitor).\n")
print("wrote", len(html) // 1024, "KB")
