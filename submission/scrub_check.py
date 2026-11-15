"""Scan files intended for publication for anything that should not leave this machine.

usage: scrub_check.py ROOT [ROOT ...] [--fix-json]

Reports local absolute paths, usernames, emails and account handles. With --fix-json it rewrites JSON
result files in place, replacing machine paths with placeholders. Code files are only reported, never
edited automatically, because a default path may be load-bearing.
"""
import json, os, re, sys

# Local identifiers come from the environment, so this file does not itself contain the names it strips.
# DRIFT_SCRUB_TERMS: comma-separated, e.g. your OS username and any account handles.
_TERMS = [t.strip() for t in os.environ.get("DRIFT_SCRUB_TERMS", "").split(",") if t.strip()]

PATTERNS = [
    ("windows_drive_path", re.compile(r"[A-Za-z]:[\\/][A-Za-z0-9_\\/.\- ]{3,}")),
    ("home_path", re.compile(r"(?i)(/home/[a-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|%USERPROFILE%|~[\\/])")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("ssh_key_name", re.compile(r"(?i)(?:codex_)?vast_\w+|id_(?:rsa|ed25519)\w*")),
    ("api_key_like", re.compile(r"\b(?:sk|pk|ghp|hf)_[A-Za-z0-9]{12,}")),
    ("long_hex_secret", re.compile(r"\b[0-9a-f]{56,}\b")),
]
if _TERMS:
    PATTERNS.append(("local_identifier",
                     re.compile(r"(?i)\b(?:" + "|".join(re.escape(t) for t in _TERMS) + r")\w*")))

# Public identifiers that are meant to be in the write-up.
ALLOW = re.compile(r"(?i)(CohereLabs/|Lexsi-Labs/|huggingface\.co|github\.com/Lexsi|support@lexsi\.ai|"
                   r"auditkit\.lexsi\.ai|hackathon\.lexsi\.ai|example\.(com|org)|@x\.com|@y\.com|"
                   r"\.example\b|acme-public|0\.0\.0\.0|127\.0\.0\.1)")

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".uv", ".pytest_cache"}
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".jinja", ".toml", ".cfg", ".sh"}

REPLACEMENTS = [
    (re.compile(r"(?i)[A-Za-z]:[\\/]SyncedProjects[\\/]Research[\\/]Drift"), "<PROJECT_ROOT>"),
    (re.compile(r"(?i)[A-Za-z]:[\\/]Data[\\/]Drift"), "<DATA_ROOT>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<host>"),
    (re.compile(r"(?i)(?:codex_)?vast_\w+"), "<ssh-key>"),
] + ([(re.compile(r"(?i)\b(?:" + "|".join(re.escape(t) for t in _TERMS) + r")\w*"), "<user>")] if _TERMS else [])


def scan_text(text):
    hits = []
    for name, pat in PATTERNS:
        for m in pat.finditer(text):
            s = m.group(0)
            if ALLOW.search(s):
                continue
            hits.append((name, s[:90]))
    return hits


def main():
    roots = [a for a in sys.argv[1:] if not a.startswith("--")]
    fix = "--fix-json" in sys.argv
    total, files_with_hits, fixed = 0, 0, 0
    summary = {}

    for root in roots:
        if os.path.isfile(root):
            paths = [root]
        else:
            paths = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in TEXT_EXT:
                        paths.append(os.path.join(dirpath, fn))

        for p in paths:
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            # for very large data files, only scan a sample
            sample = text if len(text) < 2_000_000 else text[:2_000_000]
            hits = scan_text(sample)
            if not hits:
                continue
            files_with_hits += 1
            total += len(hits)
            kinds = {}
            for k, s in hits:
                kinds.setdefault(k, set()).add(s)
            summary[p] = {k: sorted(v)[:4] for k, v in kinds.items()}
            print(f"\n{p}")
            for k, vals in sorted(kinds.items()):
                shown = "; ".join(sorted(vals)[:3])
                more = f" (+{len(vals)-3} more)" if len(vals) > 3 else ""
                print(f"  {k:20} {len(vals):4}  e.g. {shown[:110]}{more}")

            if fix and p.lower().endswith((".json", ".txt")):
                new = text
                for pat, rep in REPLACEMENTS:
                    new = pat.sub(rep, new)
                if new != text:
                    open(p, "w", encoding="utf-8").write(new)
                    fixed += 1
                    print(f"  -> rewritten with placeholders")

    print(f"\n{total} matches across {files_with_hits} files"
          + (f"; {fixed} rewritten" if fix else "; nothing modified (report only)"))
    if summary:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scrub_report.json")
        json.dump(summary, open(out, "w"), indent=1)
        print(f"report: {out}")


if __name__ == "__main__":
    main()
