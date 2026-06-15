"""Dump a few raw batch responses to diagnose parse failures."""
import json
import sys

import requests

import const

slug = sys.argv[1] if len(sys.argv) > 1 else "DeepSeekZS"
op = json.load(open(f"_batch_op_{slug}.json"))["op"]
r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{op}",
                 params={"key": const.GEMINI_API_KEY}, timeout=120)
j = r.json()
inlined = (j.get("response", {}).get("inlinedResponses", {})
           .get("inlinedResponses", []))
print("total:", len(inlined))
for item in inlined[:4]:
    print("=" * 60)
    print("meta:", item.get("metadata"))
    if "error" in item:
        print("ERROR:", json.dumps(item["error"])[:300])
        continue
    resp = item.get("response", {})
    for cand in resp.get("candidates", [])[:1]:
        print("finishReason:", cand.get("finishReason"))
        parts = cand.get("content", {}).get("parts", [])
        print("n_parts:", len(parts))
        for p in parts:
            keys = list(p.keys())
            print("part keys:", keys, "| thought:", p.get("thought"))
            t = p.get("text", "")
            print("text head:", repr(t[:200]))
            print("text tail:", repr(t[-200:]))
