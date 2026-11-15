"""Check that each reasoning task's check.py passes on a known-correct solution. usage: iq_verify.py TASKS_HOLDOUT_DIR"""
import os, shutil, subprocess, sys, tempfile

root = sys.argv[1]
Q = chr(34)
CSV = '''def parse_line(line):
    out, cur, in_q, was_q, i, n = [], [], False, False, 0, len(line)

    def flush():
        s = "".join(cur)
        out.append(s if was_q else s.strip(" \\t"))
    while i < n:
        ch = line[i]
        if in_q:
            if ch == QUOTE:
                if i + 1 < n and line[i + 1] == QUOTE:
                    cur.append(QUOTE); i += 1
                else:
                    in_q = False
            else:
                cur.append(ch)
        elif ch == QUOTE and not "".join(cur).strip(" \\t"):
            cur.clear(); in_q = True; was_q = True
        elif ch == ",":
            flush(); cur.clear(); was_q = False
        elif was_q and ch in " \\t":
            pass
        else:
            cur.append(ch)
        i += 1
    flush()
    return out
'''.replace("QUOTE", "chr(34)")
SOL = {
    "iq-slots": {"slots.py": '''def free_slots(busy, day_start, day_end):
    out, prev = [], day_start
    for s, e in sorted(busy):
        s, e = max(s, day_start), min(e, day_end)
        if e <= s:
            continue
        if s > prev:
            out.append((prev, s))
        prev = max(prev, e)
    if prev < day_end:
        out.append((prev, day_end))
    return out
'''},
    "iq-split": {"bill.py": '''def split_bill(total_cents, weights):
    s = sum(weights)
    base = [total_cents * w // s for w in weights]
    rem = [total_cents * w % s for w in weights]
    for i in sorted(range(len(weights)), key=lambda i: (-rem[i], i))[:total_cents - sum(base)]:
        base[i] += 1
    return base
'''},
    "iq-puzzle": {"schedule.txt": "1: E\n2: B\n3: D\n4: A\n5: C\n"},
    "iq-outage": {"rootcause.txt": "cause: c-88\nThe pool size was cut from 40 to 4.\n"},
    "iq-csvline": {"rowparse.py": CSV},
    "iq-order": {"order.txt": "core\ndb\ndocs\nnet\nauth\ncache\napi\ncli\nweb\n"},
}
for t, files in SOL.items():
    ws = tempfile.mkdtemp(prefix="drift-iq-")
    shutil.copytree(os.path.join(root, t, "files"), ws, dirs_exist_ok=True)
    for rel, c in files.items():
        open(os.path.join(ws, rel), "w", encoding="utf-8", newline="\n").write(c)
    r = subprocess.run([sys.executable, "-I", os.path.join(root, t, "check.py"), ws], capture_output=True, text=True)
    print(t, "PASS" if r.returncode == 0 else "FAIL " + r.stderr[-400:])
