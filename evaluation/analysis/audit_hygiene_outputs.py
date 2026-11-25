"""Quick audit: count hygiene fix markers in chap 6-10 hygiene outputs vs TM_on baseline.

Reports whether the corrections we applied actually surface in the translations.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANS_DIR = ROOT / "data" / "LOTM" / "translations"

CJK_RE = re.compile(r"[㐀-鿿]")

MARKERS = {
    # (name, regex_pattern, "good" or "bad")
    "Zhou Mingrui (pinyin)": ("Zhou Mingrui", "bad"),
    "Chu Minh Thụy (han-viet)": ("Chu Minh Thụy", "good"),
    "Kẻ Khờ (Fool fix)": ("Kẻ Khờ", "good"),
    "Ông Đùa (old Fool)": ("Ông Đùa", "bad"),
    "phi phàm (Beyonder fix)": (r"\bphi phàm\b|\bPhi Phàm\b", "good"),
    "Người Bất Tận (old)": ("Người Bất Tận", "bad"),
    "Mây Tăm (old gray fog)": ("Mây Tăm", "bad"),
    "sương mù xám (new)": ("sương mù xám", "good"),
    "Đội Tàng Xà (old)": ("Đội Tàng Xà", "bad"),
    "Kẻ Gác Đêm (new Nighthawks)": ("Kẻ Gác Đêm", "good"),
    "ripples (English leak)": (r"\bripples?\b", "bad"),
    "Whoosh/Swoosh (En leak)": (r"\b(Whoosh|Swoosh|swoosh|whoosh)\b", "bad"),
}


def count_matches(path: Path, pattern: str) -> int:
    if not path.exists():
        return -1
    text = path.read_text(encoding="utf-8")
    return len(re.findall(pattern, text))


def count_cjk(path: Path) -> int:
    if not path.exists():
        return -1
    return len(CJK_RE.findall(path.read_text(encoding="utf-8")))


def main():
    for chap in [6, 7, 8, 9, 10]:
        tm_path = TRANS_DIR / f"trans_chap_{chap}_qwen3-8b-local_yes_tm.txt"
        hy_path = TRANS_DIR / f"trans_chap_{chap}_qwen3-8b-local_yes_hygiene.txt"
        print(f"\n=== chap {chap}  (TM_on vs hygiene_on) ===")
        print(f"  {'marker':<35} {'TM_on':>7}  {'hygiene':>9}  delta")
        for name, (pattern, polarity) in MARKERS.items():
            tm_n = count_matches(tm_path, pattern)
            hy_n = count_matches(hy_path, pattern)
            delta = hy_n - tm_n
            arrow = " (good)" if (delta > 0 and polarity == "good") or (delta < 0 and polarity == "bad") else (" (bad)" if delta != 0 else "")
            print(f"  {name:<35} {tm_n:>7d}  {hy_n:>9d}  {delta:+d}{arrow}")
        tm_cjk = count_cjk(tm_path)
        hy_cjk = count_cjk(hy_path)
        cjk_delta = hy_cjk - tm_cjk
        cjk_label = " (good)" if cjk_delta < 0 else (" (bad)" if cjk_delta > 0 else "")
        print(f"  {'CJK chars':<35} {tm_cjk:>7d}  {hy_cjk:>9d}  {cjk_delta:+d}{cjk_label}")


if __name__ == "__main__":
    main()
