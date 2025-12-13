import time
import re
from abc import ABC, abstractmethod
from utils import split_text_with_overlap, extract_json_from_response, get_spacy_candidates, filter_similar_names, has_chinese_char
from modules.prompts import build_extraction_prompt, build_glossary_extraction_prompt, build_translation_prompt
import const

class BasePipeline(ABC):
    def __init__(self, llm_backend, graph_manager):
        self.llm = llm_backend
        self.gm = graph_manager

    @abstractmethod
    def process_chapter(self, text, chap_idx):
        pass

    def call_llm_with_retry(self, messages, temp, check_chinese=True):
        # Logic Retry chung cho cả 2 pipeline
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(messages, temperature=temp)
                # if check_chinese and has_chinese_char(response):
                #     print(response)
                #     print(f"   ⚠️ Detect Chinese characters, retrying ({attempt+1}/{max_retries})...")
                #     continue
                return response
            except Exception as e:
                print(f"   ⚠️ API Error: {e}, retrying...")
                time.sleep(1)
        raise RuntimeError("LLM retry failed after maximum attempts.")

class GraphRAGPipeline(BasePipeline):
    def process_chapter(self, text, chap_idx):
        print(f"🚀 RUNNING: GraphRAG Pipeline for Chapter {chap_idx}")
        
        # --- PHASE 1: MINING (Only if new chapter) ---
        if chap_idx > self.gm.tracking["last_processed_chap"]:
            print("   🛠️  Phase 1: Knowledge Mining...")
            self._phase_1_mining(text, chap_idx)
            self.gm.update_tracking(chap_idx)
            self.gm.save_data()
        else:
            print("   ⏭️  Phase 1 Skipped (Already processed).")

        # --- PHASE 2: TRANSLATION ---
        print("   ✍️  Phase 2: Translation with Graph Context...")
        translation = self._phase_2_translation(text, chap_idx)
        return translation

    def _phase_1_mining(self, full_text, chap_idx):
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE)
        current_glossary = {}
        
        for idx, chunk in enumerate(chunks):
            print(f"\t Đang mine chunk thứ {idx+1}/{len(chunks)} của chapter {chap_idx}")
            # 1. Mining Nodes/Edges
            candidates = filter_similar_names(get_spacy_candidates(chunk))
            existing = [n for n in self.gm.get_existing_node_ids() if n in candidates]
            
            prompt = build_extraction_prompt(chunk, existing, candidates)
            response = self.call_llm_with_retry(prompt, const.TEMP_MINING)
            print("----- Response -------- \n\n\n\n\n", response)
            updates = extract_json_from_response(response)
            
            if updates:
                self.gm.update_graph_state(updates, chap_idx, candidates)

            # 2. Mining Glossary
            gloss_prompt = build_glossary_extraction_prompt(chunk)
            gloss_resp = self.call_llm_with_retry(gloss_prompt, 0.05, check_chinese=True)
            gloss_data = extract_json_from_response(gloss_resp)
            if gloss_data and "glossary" in gloss_data:
                current_glossary.update(gloss_data["glossary"])

        # Update global stats
        self.gm.update_node_statistics(full_text, chap_idx)
        self.gm.update_glossary(current_glossary)
        
        # Summarize
        summary_prompt = [{"role": "user", "content": f"Summarize this chapter in Vietnamese (max 50 words):\n\n{full_text[:30000]}"}]
        summary = self.call_llm_with_retry(summary_prompt, const.TEMP_MINING)
        self.gm.add_history(chap_idx, summary)

    def _phase_2_translation(self, full_text, chap_idx):
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE, 0)
        full_trans = ""
        
        # Get context
        summary = next((h['summary'] for h in self.gm.local_graph['history'] if h['chapter'] == chap_idx), "")
        
        for idx, chunk in enumerate(chunks):
            print(f"\t Đang dịch chunk thứ {idx+1}/{len(chunks)} của chapter {chap_idx}")
            relations = self.gm.get_relations_string(chunk)
            prompt = build_translation_prompt(chunk, self.gm.glossary, relations, summary, model_type=const.MODEL_TYPE)
            
            response = self.call_llm_with_retry(prompt, const.TEMP_CREATIVE)
            
            # Post-process (Extract translation only)
            parts = re.split(r"###\s*BẢN DỊCH", response, flags=re.IGNORECASE)
            trans_text = parts[1].strip() if len(parts) > 1 else response
            full_trans += trans_text + "\n\n"
            
        return full_trans

class SimplePipeline(BasePipeline):
    def process_chapter(self, text, chap_idx):
        print(f"🚀 RUNNING: Simple (No-RAG) Pipeline for Chapter {chap_idx}")
        # Không Mining, Không Context Graph
        
        chunks = split_text_with_overlap(text, const.CHUNK_SIZE, 0)
        full_trans = ""
        
        for chunk in chunks:
            # Chỉ truyền chunk text. 
            # Các tham số glossary, relations_str, current_summary sẽ mặc định là None -> Kích hoạt Non-RAG Mode
            prompt = build_translation_prompt(chunk) 
            
            response = self.call_llm_with_retry(prompt, const.TEMP_CREATIVE)
            
            parts = re.split(r"###\s*BẢN DỊCH", response, flags=re.IGNORECASE)
            trans_text = parts[1].strip() if len(parts) > 1 else response
            full_trans += trans_text + "\n\n"
            
        return full_trans