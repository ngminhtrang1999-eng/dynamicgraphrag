"""One-shot alignment for A3 ablation runs.

Aligns source + ref + two candidate translation files (TM_on vs TM_off) into
a single aligned_master_chap_<N>.csv with columns:
    Source_ID, Source_Text, ref, TM_on, TM_off

Drops the multi-process complexity of batch_alignment.py; this is intended for
the small ablation set (chap 6, later 7-10).

Usage:
    python align_ablation.py \\
        --chap 6 \\
        --src data/LOTM/chapters/chapter_0006.txt \\
        --ref data/LOTM/translations/reference/ref_6.txt \\
        --tm-on  data/LOTM/translations/trans_chap_6_qwen3-8b-local_yes_tm.txt \\
        --tm-off data/LOTM/translations/trans_chap_6_qwen3-8b-local_yes_notm.txt \\
        --out-dir evaluation/alignment_results_qwen3-8b-local-ablation
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/LaBSE"


def smart_sentence_split(text):
    text = str(text).replace("…", "...").replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    pattern = r'(?<=[.?!])\s+(?=[A-Z"“0-9])'
    return [s.strip() for s in re.split(pattern, text) if len(s.strip()) > 1]


def global_dynamic_alignment(src_list, tgt_list, model, device,
                             max_merge_tgt=5, max_merge_src=5):
    n, m = len(src_list), len(tgt_list)
    src_emb = F.normalize(
        model.encode(src_list, convert_to_tensor=True, device=device, show_progress_bar=False),
        p=2, dim=1,
    )
    tgt_emb = F.normalize(
        model.encode(tgt_list, convert_to_tensor=True, device=device, show_progress_bar=False),
        p=2, dim=1,
    )
    dp = np.full((n + 1, m + 1), -np.inf)
    dp[0][0] = 0
    trace = {}
    PEN_T, PEN_S = 0.05, 0.1
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == -np.inf:
                continue
            if i < n:
                acc = torch.zeros_like(tgt_emb[0])
                for k in range(1, max_merge_tgt + 1):
                    if j + k > m:
                        break
                    acc = acc + tgt_emb[j + k - 1]
                    score = torch.dot(src_emb[i], F.normalize(acc, p=2, dim=0)).item()
                    new_score = dp[i][j] + score - PEN_T * (k - 1)
                    if new_score > dp[i + 1][j + k]:
                        dp[i + 1][j + k] = new_score
                        trace[(i + 1, j + k)] = (i, j, f"1-{k}")
            if j < m:
                acc = torch.zeros_like(src_emb[0])
                for k in range(1, max_merge_src + 1):
                    if i + k > n:
                        break
                    acc = acc + src_emb[i + k - 1]
                    score = torch.dot(F.normalize(acc, p=2, dim=0), tgt_emb[j]).item()
                    new_score = dp[i][j] + score - PEN_S * (k - 1)
                    if new_score > dp[i + k][j + 1]:
                        dp[i + k][j + 1] = new_score
                        trace[(i + k, j + 1)] = (i, j, f"{k}-1")
    ci, cj = n, m
    if (ci, cj) not in trace:
        best = -np.inf
        for x in range(n + 1):
            if dp[x][m] > best:
                best, ci, cj = dp[x][m], x, m
        for y in range(m + 1):
            if dp[n][y] > best:
                best, ci, cj = dp[n][y], n, y
    pairs = []
    while ci > 0 and cj > 0:
        if (ci, cj) not in trace:
            if ci > 0:
                ci -= 1
            elif cj > 0:
                cj -= 1
            continue
        pi, pj, mtype = trace[(ci, cj)]
        pairs.append({
            "Ref": " ".join(src_list[pi:ci]),
            "Trans": " ".join(tgt_list[pj:cj]),
            "Type": mtype,
        })
        ci, cj = pi, pj
    pairs.reverse()
    return pairs


def fill_column(src_sents, pairs):
    """Anchor each aligned pair to its first source sentence; mark merged rows."""
    col = [None] * len(src_sents)
    si = 0
    pi = 0
    while si < len(src_sents) and pi < len(pairs):
        p = pairs[pi]
        num_src = int(p["Type"].split("-")[0])
        if si < len(col):
            col[si] = p["Trans"]
        for k in range(1, num_src):
            if si + k < len(col):
                col[si + k] = "<MERGED_ABOVE>"
        si += num_src
        pi += 1
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chap", type=int, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tm-on", required=True)
    ap.add_argument("--tm-off", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    for p in [args.src, args.ref, args.tm_on, args.tm_off]:
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 Loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME).to(device)

    def read(p):
        with open(p, "r", encoding="utf-8-sig") as f:
            return f.read()

    src_sents = smart_sentence_split(read(args.src))
    ref_sents = smart_sentence_split(read(args.ref))
    tm_on_sents = smart_sentence_split(read(args.tm_on))
    tm_off_sents = smart_sentence_split(read(args.tm_off))
    print(f"  source: {len(src_sents)} sents | ref: {len(ref_sents)} | TM_on: {len(tm_on_sents)} | TM_off: {len(tm_off_sents)}")

    print("Aligning ref...")
    p_ref = global_dynamic_alignment(src_sents, ref_sents, model, device)
    print("Aligning TM_on...")
    p_on = global_dynamic_alignment(src_sents, tm_on_sents, model, device)
    print("Aligning TM_off...")
    p_off = global_dynamic_alignment(src_sents, tm_off_sents, model, device)

    df = pd.DataFrame(
        {
            "Source_ID": list(range(len(src_sents))),
            "Source_Text": src_sents,
            "ref": fill_column(src_sents, p_ref),
            "TM_on": fill_column(src_sents, p_on),
            "TM_off": fill_column(src_sents, p_off),
        }
    )
    out_path = os.path.join(args.out_dir, f"aligned_master_chap_{args.chap}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {out_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
