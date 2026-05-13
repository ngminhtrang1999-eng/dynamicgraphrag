"""Probe free judge providers for remaining quota.

Sends a minimal 'reply with the word OK' prompt to each candidate provider/model.
Reports which ones respond, which hit rate limits, which have other errors.

Constraint: judge family should differ from translators (llama/qwen/gemma).
Acceptable families: DeepSeek, GLM/z-ai, Gemini, Mistral, Cohere, Claude.

Usage:
    python probe_judge_providers.py
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def probe_openai_compat(label: str, base_url: str, api_key_env: str, model: str) -> tuple[str, str]:
    """Return (status, detail). status in {'OK', 'RATE_LIMIT', 'AUTH', 'OTHER_ERR'}."""
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        return "NO_KEY", f"env {api_key_env} not set"

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
            temperature=0,
            max_tokens=10,
        )
        content = (resp.choices[0].message.content or "").strip()
        return "OK", content[:80]
    except Exception as e:
        msg = str(e)
        if "429" in msg or "rate" in msg.lower() or "limit" in msg.lower():
            return "RATE_LIMIT", msg[:200]
        if "401" in msg or "auth" in msg.lower():
            return "AUTH", msg[:200]
        return "OTHER_ERR", msg[:200]


def probe_gemini(model: str = "gemini-2.5-flash") -> tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "NO_KEY", "GEMINI_API_KEY not set"
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(
            "Reply with exactly one word: OK",
            generation_config={"temperature": 0, "max_output_tokens": 10},
        )
        return "OK", (resp.text or "")[:80]
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
            return "RATE_LIMIT", msg[:200]
        if "401" in msg or "403" in msg or "permission" in msg.lower():
            return "AUTH", msg[:200]
        return "OTHER_ERR", msg[:200]


def main():
    # Family policy:
    #   AVOID llama/qwen/gemma judges (same family as translators in this project).
    #   PREFER deepseek/glm/mistral/cohere/gemini-pro (different family).
    # All candidates are free-tier or pay-as-you-go with a free credit allowance.
    probes = [
        # OpenRouter free models — each model has its own daily limit, different
        # from the global per-account 50/day. Different families only.
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
         "deepseek/deepseek-chat-v3-0324:free"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
         "deepseek/deepseek-r1:free"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
         "mistralai/mistral-small-3.1-24b-instruct:free"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
         "z-ai/glm-4.6:free"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
         "z-ai/glm-4.5-air:free"),  # known exhausted, for confirmation
        # NVIDIA NIM — hosts many families incl. deepseek; check whether deepseek is available
        ("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
         "deepseek-ai/deepseek-r1"),
        # LangDB — gateway; deepinfra/deepseek may work
        ("langdb", "LangDB", "https://api.us-east-1.langdb.ai", "LANGDB_API_KEY",
         "deepinfra/deepseek-ai/DeepSeek-V3"),
    ]

    results = []
    for kind, label, base_url, key_env, model in probes:
        print(f"[probe] {label:<12}  {model}")
        if kind == "openrouter" or kind == "nvidia" or kind == "langdb":
            status, detail = probe_openai_compat(label, base_url, key_env, model)
        else:
            status, detail = "SKIP", "unknown kind"
        marker = {"OK": "OK", "RATE_LIMIT": "RL", "AUTH": "AUTH", "NO_KEY": "NOKEY",
                  "OTHER_ERR": "ERR"}.get(status, "?")
        results.append((marker, label, model, status, detail))
        print(f"   -> [{marker}] {status}  {detail[:120]}")
        time.sleep(1.5)

    # Gemini direct
    print(f"[probe] Gemini       gemini-2.5-flash")
    status, detail = probe_gemini("gemini-2.5-flash")
    marker = {"OK": "OK", "RATE_LIMIT": "RL", "AUTH": "AUTH", "NO_KEY": "NOKEY",
              "OTHER_ERR": "ERR"}.get(status, "?")
    results.append((marker, "Gemini", "gemini-2.5-flash", status, detail))
    print(f"   -> [{marker}] {status}  {detail[:120]}")

    print()
    print("=" * 72)
    print(f"{'flag':<6} {'provider':<14} {'model':<48} status")
    print("-" * 72)
    for marker, label, model, status, detail in results:
        print(f"{marker:<6} {label:<14} {model:<48} {status}")
    print("=" * 72)
    available = [r for r in results if r[3] == "OK"]
    print(f"\nUsable now: {len(available)}/{len(results)}")
    for marker, label, model, status, _ in available:
        print(f"  - {label}: {model}")


if __name__ == "__main__":
    main()
