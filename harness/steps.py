"""Normalise a recorded session into ordered items and assistant steps.

load(session_dir) -> (meta, items)
item: {"n": order, "t": seconds, "kind": user|hidden_user|thinking|text|tool_call|tool_result, "step": k or None, "content": str, ...}
A step is one assistant message plus the tool results it produced.
"""
import json, os, sys


def _txt(c):
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") for b in c if b.get("type") == "text")


def load(sdir):
    meta = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
    hidden = {i["text"] for i in meta["injected"] if i["kind"] == "nudge"}
    advisor = {i["text"] for i in meta["injected"] if i["kind"] == "advisor"}
    return meta, parse(open(os.path.join(sdir, "events.jsonl"), encoding="utf-8"), hidden, advisor)


def parse(events, hidden=(), advisor=()):
    """events: JSON lines or already-decoded event dicts. advisor: texts of closed-loop supervisor notes."""
    items, step, call_step = [], 0, {}
    for line in events:
        e = json.loads(line) if isinstance(line, str) else line
        if e.get("type") != "message_end":
            continue
        m = e["message"]
        role = m.get("role")
        base = {"t": e["_t"], "ev": e["_i"]}
        if role == "user":
            text = _txt(m.get("content"))
            kind = "hidden_user" if text in hidden else "advisor" if text in advisor else "user"
            items.append(dict(base, kind=kind, step=None, content=text))
        elif role == "assistant":
            if m.get("stopReason") == "error":
                continue
            step += 1
            for b in m.get("content") or []:
                bt = b.get("type")
                if bt == "thinking":
                    if b.get("thinking"):
                        items.append(dict(base, kind="thinking", step=step, content=b["thinking"], redacted=bool(b.get("redacted"))))
                elif bt == "text":
                    if b.get("text", "").strip():
                        items.append(dict(base, kind="text", step=step, content=b["text"]))
                elif bt == "toolCall":
                    call_step[b.get("id")] = step
                    items.append(dict(base, kind="tool_call", step=step, tool=b.get("name"), content=json.dumps(b.get("arguments") or {}, ensure_ascii=False)))
        elif role == "toolResult":
            items.append(dict(base, kind="tool_result", step=call_step.get(m.get("toolCallId"), step), tool=m.get("toolName"),
                              is_error=bool(m.get("isError")), content=_txt(m.get("content"))))
    for n, it in enumerate(items):
        it["n"] = n
    return items


def render(items, clip=400, show_hidden=True, tiers=("thinking", "text", "tool_call", "tool_result")):
    out = []
    for it in items:
        k = it["kind"]
        if k == "hidden_user" and not show_hidden:
            continue
        if k not in ("user", "hidden_user", "advisor") and k not in tiers:
            continue
        c = it["content"].strip()
        if len(c) > clip:
            c = c[:clip] + f" ...[+{len(c) - clip} chars]"
        tag = {"user": "USER", "hidden_user": "HIDDEN-INSTRUCTION", "advisor": "ADVISOR-NOTE", "thinking": "THINK", "text": "SAY",
               "tool_call": f"CALL {it.get('tool')}", "tool_result": f"RESULT{' (error)' if it.get('is_error') else ''}"}[k]
        out.append(f"[{'s' + str(it['step']) if it['step'] else '-'}] {tag}: {c}")
    return "\n".join(out)


if __name__ == "__main__":
    meta, items = load(sys.argv[1])
    print(meta["session_id"], "|", meta["condition"], "| check:", meta["check"] and meta["check"]["passed"], "| changed:", meta["files_changed"])
    print(render(items, clip=int(sys.argv[2]) if len(sys.argv) > 2 else 260))
