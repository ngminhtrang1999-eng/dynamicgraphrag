"""Pairwise judging via the Gemini Batch API (50% cost vs sync, separate quota).

Builds the same two-call position-swap protocol as batch_pairwise_judge.py but
submits all calls as inline batch requests, polls until the batch completes,
reconciles swapped verdicts, and writes the same per-chapter pairwise CSV
schema so analyze_new_judgments.py works unchanged.

Usage (one comparison per invocation):
    python batch_judge_gemini.py --dir evaluation/alignment_results_deepseek_zs \\
        --col-a DeepSeek_ZS --slug DeepSeekZS --chapters 6 7 8 9 10 20 ... 30
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests

import const
from modules.prompts import build_literary_pairwise_judge_prompt

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "models/gemini-3.5-flash"
OUT_DIR = Path("evaluation/llm_judge")
SKIP = {"<merged_above>", "<empty>", ""}


def messages_to_gemini(messages):
    system = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    req = {"contents": contents,
           "generation_config": {"temperature": 0.1, "max_output_tokens": 2048}}
    if system:
        req["system_instruction"] = {"parts": [{"text": system}]}
    return req


def parse_choice(raw):
    if not raw:
        return None, ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            c = (obj.get("choice") or "").strip().lower()
            if c in {"a", "b", "tie"}:
                return c, (obj.get("rationale") or "")[:500]
        except json.JSONDecodeError:
            pass
    # salvage truncated JSON (MAX_TOKENS cuts mid-rationale; choice leads)
    m = re.search(r'"choice"\s*:\s*"(a|b|tie)"', cleaned, flags=re.IGNORECASE)
    if m:
        r = re.search(r'"rationale"\s*:\s*"(.*)', cleaned, flags=re.DOTALL)
        return m.group(1).lower(), (r.group(1)[:500] if r else "")
    return None, raw[:200]


def unswap(c):
    return {"a": "b", "b": "a"}.get(c, c)


def reconcile(c1, c2u):
    if c1 is None or c2u is None:
        return None
    if c1 == c2u and c1 in {"a", "b"}:
        return c1
    if c1 == "tie" and c2u == "tie":
        return "tie"
    return "tie"


def load_done(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8-sig") as f:
        return {(r["chapter"], r["sentence_id"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--col-a", required=True)
    ap.add_argument("--col-b", default="GG")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--chapters", type=int, nargs="+", required=True)
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()

    key = const.GEMINI_API_KEY
    requests_payload = []
    pair_meta = {}  # key -> (chapter, sid)

    for n in args.chapters:
        csv_path = Path(args.dir) / f"aligned_master_chap_{n}.csv"
        out_path = OUT_DIR / f"pairwise_chap{n}_{args.slug}_vs_GG_gemini35flash.csv"
        done = load_done(out_path)
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                sid = str(r.get("Source_ID", "")).strip()
                if (str(n), sid) in done:
                    continue
                src = (r.get("Source_Text") or "").strip()
                ref = (r.get("ref") or "").strip()
                a = (r.get(args.col_a) or "").strip()
                b = (r.get(args.col_b) or "").strip()
                if not src or src.lower() in SKIP:
                    continue
                if a.lower() in SKIP or b.lower() in SKIP or not a or not b:
                    continue
                if ref.lower() in SKIP:
                    ref = ""
                for call, (x, y) in {"1": (a, b), "2": (b, a)}.items():
                    k = f"{n}|{sid}|{call}"
                    msgs = build_literary_pairwise_judge_prompt(src, x, y, reference_text=ref or None)
                    requests_payload.append({"request": messages_to_gemini(msgs),
                                             "metadata": {"key": k}})
                pair_meta[f"{n}|{sid}"] = (str(n), sid)

    if not requests_payload:
        print("nothing to judge (all done)")
        return
    print(f"{args.slug}: {len(requests_payload)} batch requests ({len(pair_meta)} pairs)")

    body = {"batch": {"display_name": f"judge_{args.slug}",
                      "input_config": {"requests": {"requests": requests_payload}}}}
    r = requests.post(f"{BASE}/{MODEL}:batchGenerateContent", params={"key": key},
                      json=body, timeout=300)
    r.raise_for_status()
    op = r.json()["name"]
    print("batch operation:", op)
    Path(f"_batch_op_{args.slug}.json").write_text(json.dumps({"op": op}), encoding="utf-8")

    while True:
        time.sleep(args.poll)
        rr = requests.get(f"{BASE}/{op}", params={"key": key}, timeout=120)
        rr.raise_for_status()
        j = rr.json()
        meta = j.get("metadata", {})
        state = meta.get("state", "?")
        print("state:", state)
        if j.get("done"):
            break

    if "error" in j:
        raise SystemExit(f"batch failed: {j['error']}")

    inlined = (j.get("response", {}).get("inlinedResponses", {})
               .get("inlinedResponses", []))
    print(f"responses: {len(inlined)}")
    verdicts = {}  # key -> (choice, rationale)
    for item in inlined:
        k = item.get("metadata", {}).get("key", "")
        resp = item.get("response", {})
        text = ""
        for cand in resp.get("candidates", [])[:1]:
            for part in cand.get("content", {}).get("parts", []):
                text += part.get("text", "")
        verdicts[k] = parse_choice(text)

    # reconcile + write per-chapter CSVs
    rows_by_chap = {}
    for pk, (chap, sid) in pair_meta.items():
        c1, r1 = verdicts.get(f"{pk}|1", (None, ""))
        c2, r2 = verdicts.get(f"{pk}|2", (None, ""))
        final = reconcile(c1, unswap(c2))
        if final is None:
            continue
        rows_by_chap.setdefault(chap, []).append({
            "chapter": chap, "sentence_id": sid,
            "call_1_choice": c1 or "", "call_2_choice_unswapped": unswap(c2) or "",
            "final_choice": final, "rationale_1": r1, "rationale_2": r2,
            "judge_model": "gemini-3.5-flash(batch)",
        })

    cols = ["chapter", "sentence_id", "call_1_choice", "call_2_choice_unswapped",
            "final_choice", "rationale_1", "rationale_2", "judge_model"]
    for chap, rows in sorted(rows_by_chap.items()):
        out_path = OUT_DIR / f"pairwise_chap{chap}_{args.slug}_vs_GG_gemini35flash.csv"
        new = not out_path.exists() or out_path.stat().st_size == 0
        with open(out_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"chap {chap}: +{len(rows)} verdicts -> {out_path.name}")


if __name__ == "__main__":
    main()
