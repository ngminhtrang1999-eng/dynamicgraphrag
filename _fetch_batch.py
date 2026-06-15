"""Poll a submitted Gemini batch op until done, then parse + write pairwise CSVs.

Resumes from _batch_op_<slug>.json (written by batch_judge_gemini.py), so it
survives the submitter process dying. Pair metadata is rebuilt from the
metadata keys embedded in the batch responses themselves.

Usage:
    python _fetch_batch.py --slug DeepSeekZS [--poll 300]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import requests

import const
from batch_judge_gemini import OUT_DIR, parse_choice, reconcile, unswap

BASE = "https://generativelanguage.googleapis.com/v1beta"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out-slug", default=None,
                    help="slug used in output CSV names (default: --slug)")
    ap.add_argument("--poll", type=int, default=300)
    args = ap.parse_args()
    out_slug = args.out_slug or args.slug

    op = json.load(open(f"_batch_op_{args.slug}.json"))["op"]
    print(f"polling {op}", flush=True)
    while True:
        r = requests.get(f"{BASE}/{op}", params={"key": const.GEMINI_API_KEY},
                         timeout=120)
        r.raise_for_status()
        j = r.json()
        meta = j.get("metadata", {})
        print("state:", meta.get("state"), json.dumps(meta.get("batchStats", {})),
              flush=True)
        if j.get("done"):
            break
        time.sleep(args.poll)

    if "error" in j:
        raise SystemExit(f"BATCH_FAILED: {json.dumps(j['error'])[:800]}")

    inlined = (j.get("response", {}).get("inlinedResponses", {})
               .get("inlinedResponses", []))
    print(f"responses: {len(inlined)}", flush=True)
    verdicts = {}
    for item in inlined:
        k = item.get("metadata", {}).get("key", "")
        resp = item.get("response", {})
        text = ""
        for cand in resp.get("candidates", [])[:1]:
            for part in cand.get("content", {}).get("parts", []):
                text += part.get("text", "")
        verdicts[k] = parse_choice(text)

    pair_keys = {k.rsplit("|", 1)[0] for k in verdicts}
    rows_by_chap = {}
    skipped = 0
    for pk in pair_keys:
        chap, sid = pk.split("|", 1)
        c1, r1 = verdicts.get(f"{pk}|1", (None, ""))
        c2, r2 = verdicts.get(f"{pk}|2", (None, ""))
        final = reconcile(c1, unswap(c2))
        if final is None:
            skipped += 1
            continue
        rows_by_chap.setdefault(chap, []).append({
            "chapter": chap, "sentence_id": sid,
            "call_1_choice": c1 or "", "call_2_choice_unswapped": unswap(c2) or "",
            "final_choice": final, "rationale_1": r1, "rationale_2": r2,
            "judge_model": "gemini-3.5-flash(batch)",
        })

    cols = ["chapter", "sentence_id", "call_1_choice", "call_2_choice_unswapped",
            "final_choice", "rationale_1", "rationale_2", "judge_model"]
    for chap, rows in sorted(rows_by_chap.items(), key=lambda kv: int(kv[0])):
        rows.sort(key=lambda r: r["sentence_id"])
        out_path = OUT_DIR / f"pairwise_chap{chap}_{out_slug}_vs_GG_gemini35flash.csv"
        new = not out_path.exists() or out_path.stat().st_size == 0
        # avoid duplicate rows if fetch runs twice
        done = set()
        if not new:
            with open(out_path, encoding="utf-8-sig") as f:
                done = {(r["chapter"], r["sentence_id"]) for r in csv.DictReader(f)}
        rows = [r for r in rows if (r["chapter"], r["sentence_id"]) not in done]
        mode = "w" if new else "a"
        with open(out_path, mode, encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"chap {chap}: +{len(rows)} verdicts", flush=True)
    print(f"unparseable pairs skipped: {skipped}", flush=True)
    print(f"FETCH_DONE {args.slug} -> {out_slug}", flush=True)


if __name__ == "__main__":
    main()
