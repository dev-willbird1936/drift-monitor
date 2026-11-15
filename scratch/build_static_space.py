"""Build the static Space (free hosting): a replay viewer for submission/space/replays.json.

usage: build_static_space.py  -> writes submission/space_static/index.html and README.md
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "submission", "space_static")
reps = json.load(open(os.path.join(ROOT, "submission", "space", "replays.json"), encoding="utf-8"))
links = ('<a href="https://huggingface.co/dev-willbird1936/drift-monitor-tiny-aya">Model</a> · '
         '<a href="https://huggingface.co/datasets/dev-willbird1936/drift-monitor-data">Dataset and traces</a> · '
         '<a href="https://github.com/dev-willbird1936/drift-monitor">Code and write-up</a> · '
         'Run it yourself: <code>try/try_monitor.bat</code>, <code>submission/space/driftwatch.py</code> or the Pi extension in the repo.')
data = json.dumps(reps, ensure_ascii=False).replace("</", "<\\/")
html = open(os.path.join(OUT, "template.html"), encoding="utf-8").read().replace("__LINKS__", links).replace("__DATA__", data)
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(
    "---\ntitle: Drift Monitor\nemoji: 📉\ncolorFrom: indigo\ncolorTo: gray\nsdk: static\napp_file: index.html\npinned: false\n"
    "license: cc-by-nc-4.0\nmodels:\n- dev-willbird1936/drift-monitor-tiny-aya\ndatasets:\n- dev-willbird1936/drift-monitor-data\n"
    "short_description: Replays an AI agent being watched for drift\n---\n\n"
    "Replay viewer for the drift monitor: recorded closed-loop sessions, watched against unwatched, with the monitor's score "
    "at every step and the advisor's notes. See the [model](https://huggingface.co/dev-willbird1936/drift-monitor-tiny-aya) "
    "and the [code](https://github.com/dev-willbird1936/drift-monitor).\n")
print("wrote", len(html) // 1024, "KB")
