"""One-shot poll of a Gemini batch operation saved in _batch_op_<slug>.json."""
import json
import sys

import requests

import const

slug = sys.argv[1] if len(sys.argv) > 1 else "DeepSeekZS"
op = json.load(open(f"_batch_op_{slug}.json"))["op"]
r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{op}",
                 params={"key": const.GEMINI_API_KEY}, timeout=60)
j = r.json()
meta = j.get("metadata", {})
print("state:", meta.get("state"))
print("done:", j.get("done"))
if "error" in j:
    print("ERROR:", json.dumps(j["error"])[:500])
print("stats:", json.dumps(meta.get("batchStats", {})))
