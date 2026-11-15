"""Scan everything about to be published for identifiers, secrets and local paths.

usage: prepublish_scan.py ROOT_OR_FILE [...]
Prints hits per pattern with a few samples. Exit code 1 if any hit outside the allow-list.
"""
import os, re, sys, pathlib, collections

TERMS = [t for t in os.environ.get("DRIFT_SCRUB_TERMS", "").split(",") if t]
PATTERNS = {
    "scrub_term": re.compile("|".join(re.escape(t) for t in TERMS), re.I) if TERMS else None,
    "user_path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/c/Users/|/home/[a-z]", re.I),
    "appdata": re.compile(r"AppData", re.I),
    "brain_dir": re.compile(r"\.brain[\\/]"),
    "openrouter_key": re.compile(r"sk-or-v1-[0-9a-f]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "openai_key": re.compile(r"\bsk-(proj-)?[A-Za-z0-9]{32,}"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "github_token": re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "email_real_domains": re.compile(r"[A-Za-z0-9._%+-]+@(gmail|outlook|hotmail|yahoo|icloud|proton)\.[a-z.]+", re.I),
}
SKIP_EXT = {".safetensors", ".bin", ".png", ".jpg", ".pt", ".gz", ".pyc"}

hits = collections.defaultdict(list)
n_files = 0
for root in sys.argv[1:]:
    p0 = pathlib.Path(root)
    files = [p0] if p0.is_file() else [p for p in p0.rglob("*") if p.is_file()]
    for p in files:
        if p.suffix in SKIP_EXT or "__pycache__" in p.parts or ".git" in p.parts:
            continue
        n_files += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for name, rx in PATTERNS.items():
            if rx is None:
                continue
            for m in rx.finditer(text):
                s = max(0, m.start() - 50)
                hits[name].append((str(p), text[s:m.end() + 50].replace("\n", " ")))
print(f"scanned {n_files} files")
for name, hs in hits.items():
    files = collections.Counter(h[0] for h in hs)
    print(f"\n== {name}: {len(hs)} hits in {len(files)} files")
    for f, c in files.most_common(8):
        print(f"   {c:5}  {f}")
    for f, ctx in hs[:4]:
        print(f"   e.g. {f}: ...{ctx[:180]}...")
sys.exit(1 if hits else 0)
