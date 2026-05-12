import argparse
import os
import re
import const
from modules.graph_manager import GraphManager
from core.llm_backend import LocalQwenLLM, GeminiAPILLM, GroqLLM, OpenRouterLLM, LangDBLLM, NVIDIALLM, CloudfareLLM
from core.pipeline import GraphRAGPipeline, SimplePipeline, PostEditingPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Novel Translation System")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["qwen", "qwen3-32b" ,"gemini2.5-flash", "gemini2.5-pro", "llama3.1-8b", "llama3.3-70b", "gemma3-27B", "gemma3-12b", "llama-4-maverick-17b-128e", "qwen3-235b-a22b", "deepseek-v3.1"], 
        default="gemini2.5-pro", 
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. SETUP EXPERIMENT PATH
    # Tên thư mục sẽ là: llama3.1-8b_yes hoặc llama3.1-8b_no
    experiment_sub_dir = f"{args.model}_{args.rag}"
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
            pipeline = PostEditingPipeline(llm, gm, injection_llm=injection_llm)
        else:
            print("📌 Using GraphRAG Pipeline (Single-Pass Injection)")
            pipeline = GraphRAGPipeline(llm, gm, injection_llm=injection_llm)
    else:
        pipeline = SimplePipeline(llm, gm)

    # 5. Processing Loop
    all_files = sorted([f for f in os.listdir(const.INPUT_DIR) if f.endswith(".txt")])
    
    if not all_files:
        print(f"⚠️ Không tìm thấy file .txt nào trong {const.INPUT_DIR}")
        return

    for chap_idx in range(const.START_CHAPTER, const.END_CHAPTER + 1):
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
        # Tên file output ví dụ: trans_chap_1_llama3.1-8b_yes.txt
        out_name = f"trans_chap_{chap_idx}_{args.model}_{args.rag}.txt"
        out_path = os.path.join(const.OUTPUT_TRANS_DIR, out_name)

        # RUN PIPELINE
        # Pipeline will write incrementally to out_path
        translated_text = pipeline.process_chapter(text, chap_idx, out_path)
            
        print(f"✅ Completed Chapter {chap_idx}. Saved to {out_name}")

if __name__ == "__main__":
    main()