import argparse
import os
import re
import const
from modules.graph_manager import GraphManager
from modules.translation_memory import TranslationMemory
from core.llm_backend import LocalQwenLLM, GeminiAPILLM, GroqLLM, OpenRouterLLM, LangDBLLM, NVIDIALLM, CloudfareLLM, OllamaLLM, OpenAICompatLLM
from core.pipeline import GraphRAGPipeline, SimplePipeline, PostEditingPipeline
from modules.coreference import CoreferenceResolver

def parse_args():
    parser = argparse.ArgumentParser(description="Novel Translation System")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "qwen", "qwen3-32b", "gemini2.5-flash", "gemini2.5-pro",
            "llama3.1-8b", "llama3.3-70b", "gemma3-27B", "gemma3-12b",
            "llama-4-maverick-17b-128e", "qwen3-235b-a22b", "deepseek-v3.1",
            # Local via Ollama (thesis baseline experiments)
            "qwen3-8b-local", "llama3.1-8b-local", "gemma3-4b-local",
            # Paid API for judge / strong baseline
            "deepseek-chat",
        ],
        default="qwen3-8b-local",
        help="Choose LLM Backend"
    )
    parser.add_argument("--rag", type=str, choices=["yes", "no"], default="yes", help="Enable Graph RAG?")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["graphrag", "postedit", "simple"],
        default="postedit",
        help="Pipeline type: graphrag (old), postedit (new draft-edit), simple (no RAG)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip all input() pauses for batch runs (ablation studies, eval pipelines)",
    )
    parser.add_argument(
        "--tm",
        dest="use_tm",
        action="store_true",
        default=True,
        help="Enable Translation Memory (A3) for PostEditing draft prompt (default: on)",
    )
    parser.add_argument(
        "--no-tm",
        dest="use_tm",
        action="store_false",
        help="Disable Translation Memory (for ablation runs)",
    )
    parser.add_argument(
        "--tm-k",
        type=int,
        default=3,
        help="Number of TM examples to retrieve per chunk (default 3)",
    )
    parser.add_argument(
        "--tm-min-similarity",
        type=float,
        default=0.3,
        help="Cosine similarity floor for TM retrieval (default 0.3)",
    )
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default="",
        help="Optional suffix appended to experiment dir name (for ablation: e.g. 'tm' or 'notm')",
    )
    parser.add_argument(
        "--a2-coref",
        dest="use_a2_coref",
        action="store_true",
        default=False,
        help="Enable A2: fastcoref-based deterministic entity injection + tight glossary filter (default: off).",
    )
    parser.add_argument(
        "--no-a2-coref",
        dest="use_a2_coref",
        action="store_false",
        help="Disable A2 (use spaCy+LLM injection path).",
    )
    parser.add_argument(
        "--a2-device",
        type=str,
        default="cuda",
        help="Device for fastcoref model ('cuda' or 'cpu'); ignored when --a2-coref is off.",
    )
    parser.add_argument(
        "--chap-start",
        type=int,
        default=None,
        help="Override const.START_CHAPTER",
    )
    parser.add_argument(
        "--chap-end",
        type=int,
        default=None,
        help="Override const.END_CHAPTER",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. SETUP EXPERIMENT PATH
    # Tên thư mục: <model>_<rag>[_<suffix>] (suffix optional, dùng cho ablation)
    experiment_sub_dir = f"{args.model}_{args.rag}"
    if args.experiment_suffix:
        experiment_sub_dir = f"{experiment_sub_dir}_{args.experiment_suffix}"
    experiment_data_path = os.path.join(const.DATA_DIR, experiment_sub_dir)
    
    print(f"📂 EXPERIMENT DATA PATH: {experiment_data_path}")
    
    # Tạo thư mục riêng cho experiment này (nếu chưa có)
    os.makedirs(experiment_data_path, exist_ok=True)
    os.makedirs(const.OUTPUT_TRANS_DIR, exist_ok=True)

    # 2. Initialize Graph Manager with SPECIFIC PATH
    gm = GraphManager(experiment_data_path)
    gm.load_data()

    # 3. Factory: Initialize LLM Backend
    if args.model == "qwen":
        llm = LocalQwenLLM()
    elif args.model == "qwen3-32b":
        llm = GroqLLM(model_name="qwen/qwen3-32b")
    elif args.model == "gemini2.5-flash":
        llm = GeminiAPILLM(model_name="gemini-2.5-flash")
    elif args.model == "gemini2.5-pro":
        llm = GeminiAPILLM(model_name="gemini-2.5-pro")
    elif args.model == "llama3.1-8b":
        # [Updated] Khởi tạo Groq với model cụ thể
        llm = GroqLLM(model_name="llama-3.1-8b-instant")
    elif args.model == "llama3.3-70b":
        # [Updated] Khởi tạo Groq với model cụ thể
        llm = GroqLLM(model_name="llama-3.3-70b-versatile")
    elif args.model == "gemma3-27B":
        # [Updated] LangDB Gemma 3 27B
        # llm = LangDBLLM(model_name="deepinfra/gemma-3-27b-it")
        llm = OpenRouterLLM(model_name="google/gemma-3-27b-it:free")
    elif args.model == "gemma3-12b":
        # [Updated] OpenRouter Gemma 3 12B
        llm = OpenRouterLLM(model_name="google/gemma-3-12b-it:free")
    elif args.model == "llama-4-maverick-17b-128e":
        llm = NVIDIALLM(model_name="meta/llama-4-maverick-17b-128e-instruct")
    elif args.model == "qwen3-235b-a22b":
        # [Updated] OpenRouter
        llm = OpenRouterLLM(model_name="qwen/qwen3-235b-a22b:free")
    elif args.model == "deepseek-v3.1":
        llm = NVIDIALLM(model_name="deepseek-ai/deepseek-v3.1-terminus")
    # --- Local via Ollama ---
    elif args.model == "qwen3-8b-local":
        llm = OllamaLLM(model_name="qwen3:8b")
    elif args.model == "llama3.1-8b-local":
        llm = OllamaLLM(model_name="llama3.1:8b")
    elif args.model == "gemma3-4b-local":
        llm = OllamaLLM(model_name="gemma3:4b")
    # --- Paid API (judge / strong baseline) ---
    elif args.model == "deepseek-chat":
        llm = OpenAICompatLLM(
            model_name="deepseek-chat",
            base_url=const.DEEPSEEK_BASE_URL,
            api_key=const.DEEPSEEK_API_KEY,
        )
    else:
        raise ValueError(f"Unknown model argument: {args.model}")

    # 4. Factory: Initialize Pipeline Strategy
    if args.rag == "yes":
        # Injection Model Strategy:
        if isinstance(llm, LocalQwenLLM):
            injection_llm = llm
        else:
            print("⏳ INJECTION MODEL: Using main LLM for injection.")
            injection_llm = llm

        # Select pipeline based on --pipeline argument
        if args.pipeline == "postedit":
            print("📌 Using PostEditing Pipeline (Draft-then-Edit)")
            tm = None
            if args.use_tm:
                tm = TranslationMemory(experiment_data_path)
                if len(tm) == 0:
                    print(f"   ⚠️ TM is empty for {experiment_sub_dir}. Run seed_translation_memory.py first, or pass --no-tm.")
                else:
                    print(f"   📚 TM loaded: {len(tm)} entries from {experiment_data_path}")
            else:
                print("   🚫 TM disabled (--no-tm)")
            a2_resolver = None
            if args.use_a2_coref:
                print(f"   🔗 A2 coref: loading fastcoref on {args.a2_device}...")
                a2_resolver = CoreferenceResolver(device=args.a2_device)
                a2_resolver._ensure_loaded()  # fail fast at startup, not mid-chapter
                print(f"   ✅ A2 coref ready (device={a2_resolver.device}).")
            else:
                print("   🚫 A2 coref disabled (--no-a2-coref).")
            pipeline = PostEditingPipeline(
                llm, gm,
                injection_llm=injection_llm,
                non_interactive=args.non_interactive,
                translation_memory=tm,
                tm_k=args.tm_k,
                tm_min_similarity=args.tm_min_similarity,
                a2_resolver=a2_resolver,
            )
        else:
            print("📌 Using GraphRAG Pipeline (Single-Pass Injection)")
            pipeline = GraphRAGPipeline(llm, gm, injection_llm=injection_llm, non_interactive=args.non_interactive)
    else:
        pipeline = SimplePipeline(llm, gm, non_interactive=args.non_interactive)

    # 5. Processing Loop
    all_files = sorted([f for f in os.listdir(const.INPUT_DIR) if f.endswith(".txt")])
    
    if not all_files:
        print(f"⚠️ Không tìm thấy file .txt nào trong {const.INPUT_DIR}")
        return

    chap_start = args.chap_start if args.chap_start is not None else const.START_CHAPTER
    chap_end = args.chap_end if args.chap_end is not None else const.END_CHAPTER
    for chap_idx in range(chap_start, chap_end + 1):
        # Tìm file tương ứng chap_idx (giả sử tên file có số)
        file_name = next((f for f in all_files if str(chap_idx) in f), None)
        if not file_name: 
            continue
        
        file_path = os.path.join(const.INPUT_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Clean text cơ bản
        text = re.sub(r'\s+', ' ', text)
            
        print(f"\n====== PROCESSING CHAPTER {chap_idx} [{args.model.upper()} | RAG:{args.rag.upper()}] ======")
        
        # Save Result
        # Tên file output: trans_chap_<N>_<model>_<rag>[_<suffix>].txt
        out_name = f"trans_chap_{chap_idx}_{experiment_sub_dir}.txt"
        out_path = os.path.join(const.OUTPUT_TRANS_DIR, out_name)

        # RUN PIPELINE
        # Pipeline will write incrementally to out_path
        translated_text = pipeline.process_chapter(text, chap_idx, out_path)
            
        print(f"✅ Completed Chapter {chap_idx}. Saved to {out_name}")

if __name__ == "__main__":
    main()