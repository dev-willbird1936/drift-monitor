"""Does OpenRouter honour reasoning effort the same way for streamed and non-streamed GPT-6 Luna calls?
Prints reasoning tokens and cost for each mode. Reads the key from OPENROUTER_API_KEY."""
import json, os, urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
MSG = [{"role": "user", "content": "A train leaves at 9:40 and the trip takes 2 h 35 min with a 12 min stop. When does it arrive? Answer in one line."}]


def call(stream, effort):
    body = {"model": "openai/gpt-6-luna", "messages": MSG, "stream": stream, "max_tokens": 16000,
            "usage": {"include": True}, "reasoning": {"effort": effort}}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    usage, provider = {}, None
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            j = json.loads(r.read())
            usage, provider = j.get("usage", {}), j.get("provider")
        else:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    try:
                        j = json.loads(line[5:])
                    except ValueError:
                        continue
                    usage = j.get("usage") or usage
                    provider = j.get("provider") or provider
    det = usage.get("completion_tokens_details") or {}
    print(f"stream={stream!s:5} effort={effort:6} provider={provider} reasoning_tokens={det.get('reasoning_tokens')} "
          f"completion={usage.get('completion_tokens')} cost={usage.get('cost')}")


for effort in ("xhigh", "high"):
    for stream in (False, True):
        call(stream, effort)
