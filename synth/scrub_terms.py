"""Replace a leaked local identifier in generated data files, in place, with verification.

usage: scrub_terms.py OLD NEW FILE [FILE ...]
Each file is rewritten to FILE.tmp and swapped in only if the line count is unchanged, the old string is gone
and the number of changed lines equals the number that contained it. Prints what changed per file.
"""
import os, sys

old, new, files = sys.argv[1], sys.argv[2], sys.argv[3:]
assert old and new and old.lower() not in new.lower()
for path in files:
    if not os.path.exists(path):
        print("missing", path)
        continue
    tmp = path + ".tmp"
    n_lines = n_hit = n_repl = 0
    with open(path, encoding="utf-8", newline="") as src, open(tmp, "w", encoding="utf-8", newline="") as dst:
        for line in src:
            n_lines += 1
            if old in line:
                n_hit += 1
                n_repl += line.count(old)
                line = line.replace(old, new)
            dst.write(line)
    check_lines = left = 0
    with open(tmp, encoding="utf-8", newline="") as f:
        for line in f:
            check_lines += 1
            left += old.lower() in line.lower()
    if check_lines != n_lines or left:
        os.remove(tmp)
        print(f"NOT CHANGED {path}: lines {n_lines}->{check_lines}, still containing old (any case): {left}")
        continue
    if n_hit:
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    print(f"{path}: {n_lines} lines, {n_hit} lines changed, {n_repl} replacements")
