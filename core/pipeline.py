import time
import re
from abc import ABC, abstractmethod
from utils import split_text_with_overlap, extract_json_from_response, get_spacy_candidates, filter_similar_names, has_chinese_char, clean_entity_tags, count_tokens, add_placeholder_regex, count_sentences, get_last_sentences
from modules.prompts import build_extraction_prompt, build_glossary_extraction_prompt, build_translation_prompt, build_entity_injection_prompt, build_draft_translation_prompt, build_post_editing_prompt
import const
import spacy

# Load spacy model for sentence splitting
try:
    nlp = spacy.load("en_core_web_trf")
except:
    nlp = spacy.load("en_core_web_sm")

class BasePipeline(ABC):
    def __init__(self, llm_backend, graph_manager):
        self.llm = llm_backend
        self.gm = graph_manager

    @abstractmethod
    def process_chapter(self, text, chap_idx):
        pass

    def call_llm_with_retry(self, messages, temp, check_chinese=True, llm_engine=None):
        # Logic Retry chung cho cả 2 pipeline
        target_llm = llm_engine if llm_engine else self.llm
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = target_llm.generate(messages, temperature=temp)
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
    def __init__(self, llm_backend, graph_manager, injection_llm=None):
        super().__init__(llm_backend, graph_manager)
        self.injection_llm = injection_llm

    def process_chapter(self, text, chap_idx, out_path):
        print(f"🚀 RUNNING: GraphRAG Pipeline for Chapter {chap_idx}")
        
        # --- PHASE 1: MINING (Only if new chapter) ---
        if chap_idx > self.gm.tracking["last_processed_chap"]:
            print("   🛠️  Phase 1: Knowledge Mining...")
            self._phase_1_mining(text, chap_idx)
            self.gm.update_tracking(chap_idx)
            self.gm.save_data()
        else:
            print("   ⏭️  Phase 1 Skipped (Already processed).")

        # --- MANUAL PAUSE & RELOAD ---
        print("\n   ⏸️  Paused. You can now edit 'term_glossary.json' or 'global_graph.json'.")
        user_input = input("   Press Enter (or 'c') to continue to Phase 2, or Ctrl+C to exit: ")
        
        print("   🔄 Reloading graph data from disk...")
        self.gm.load_data()
        print("   ✅ Data reloaded.")

        # --- PHASE 2: TRANSLATION ---
        print("   ✍️  Phase 2: Translation with Graph Context...")
        translation = self._phase_2_translation(text, chap_idx, out_path)
        return translation

    def _phase_1_mining(self, full_text, chap_idx):
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE_MINING)
        current_glossary = {}
        
        for idx, chunk in enumerate(chunks):
            print(f"\t Đang mine chunk thứ {idx+1}/{len(chunks)} của chapter {chap_idx}")
            # 1. Mining Nodes/Edges
            candidates = filter_similar_names(get_spacy_candidates(chunk))
            existing = [n for n in self.gm.get_existing_node_ids() if n in candidates]
            
            prompt = build_extraction_prompt(chunk, existing, candidates)
            print(f"   ℹ️ Prompt Tokens: {count_tokens(prompt)}")
            
            # Retry loop for internal logic errors (e.g. JSON parsed as List instead of Dict)
            max_logic_retries = 3
            for attempt in range(max_logic_retries):
                response = self.call_llm_with_retry(prompt, const.TEMP_MINING)
                # print("----- Response -------- \n", response)
                updates = extract_json_from_response(response)
                
                if updates:
                    try:
                        self.gm.update_graph_state(updates, chap_idx, candidates)
                        break 
                    except Exception as e:
                        print(f"   ⚠️ Graph Update Error: {e}. Retrying ({attempt+1}/{max_logic_retries})...")
                else:
                     # updates is None (parse failed) or empty
                     if attempt < max_logic_retries - 1:
                         print(f"   ⚠️ No valid updates parsed. Retrying ({attempt+1}/{max_logic_retries})...")
                     else:
                         print("   ❌ Failed to mine updates after retries.")

            # 2. Mining Glossary
            # 2. Mining Glossary
            gloss_prompt = build_glossary_extraction_prompt(chunk)
            print(f"   ℹ️ Glossary Prompt Tokens: {count_tokens(gloss_prompt)}")
            gloss_resp = self.call_llm_with_retry(gloss_prompt, 0.05, check_chinese=True)
            gloss_data = extract_json_from_response(gloss_resp)
            if gloss_data and "glossary" in gloss_data:
                current_glossary.update(gloss_data["glossary"])

        # Update global stats
        self.gm.update_node_statistics(full_text, chap_idx)
        self.gm.update_glossary(current_glossary)
        
        # Summarize
        summary_prompt = [{"role": "user", "content": f"Summarize this chapter in English (max 35 words):\n\n{full_text}"}]
        print(f"   ℹ️ Summary Prompt Tokens: {count_tokens(summary_prompt)}")
        summary = self.call_llm_with_retry(summary_prompt, const.TEMP_MINING)
        self.gm.add_history(chap_idx, summary)

    def _phase_2_translation(self, full_text, chap_idx, out_path):
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE, 0)
        full_trans = ""
        
        # Check Tracking to Resume
        progress = self.gm.get_chapter_progress(chap_idx)
        start_chunk_idx = 0
        
        # --- Context Management ---
        prev_chunk_last_sentences = "" 

        if progress:
            saved_size = progress.get("chunk_size", 0)
            if saved_size == const.CHUNK_SIZE:
                last_done = progress.get("last_chunk_index", -1)
                start_chunk_idx = last_done + 1
                
                # LOAD CONTEXT FROM CURRENT PROGRESS
                prev_chunk_last_sentences = progress.get("last_sentences", "")

                if start_chunk_idx < len(chunks):
                    print(f"   ⏩ Resuming from chunk {start_chunk_idx+1} (Previous progress found).")
            else:
                 print(f"   ⚠️ Chunk size changed ({saved_size} -> {const.CHUNK_SIZE}). Restarting chapter.")

        # If starting from scratch (index 0), clear the output file
        if start_chunk_idx == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("") # Clear file
            
            # ATTEMPT TO LOAD CONTEXT FROM PREVIOUS CHAPTER
            if chap_idx > 1:
                prev_chap_progress = self.gm.get_chapter_progress(chap_idx - 1)
                if prev_chap_progress:
                    prev_chunk_last_sentences = prev_chap_progress.get("last_sentences", "")
                    print(f"   🔗 Context loaded from Chapter {chap_idx - 1}: '{prev_chunk_last_sentences}'")

        summary = next((h['summary'] for h in self.gm.local_graph['history'] if h['chapter'] == chap_idx), "")
        
        for idx, chunk in enumerate(chunks):
            # SKIP if already done
            if idx < start_chunk_idx:
                continue

            print(f"\t Đang dịch chunk thứ {idx+1}/{len(chunks)} của chapter {chap_idx}")
            
            # Step 1: Look-ahead NER - Extract entities from current chunk
            chunk_entities = get_spacy_candidates(chunk)
            chunk_entities = filter_similar_names(chunk_entities)
            
            # Step 2: Update active memory with found entities (Sliding Window)
            self.gm.update_active_memory(chunk_entities)
            
            # Step 3: Get relations based on active memory (not text matching)
            relations_data = self.gm.get_relations_data()
            
            # =========================================
            # NEW: Entity Injection Phase
            # =========================================
            # Step 4: Run Entity Injection to annotate pronouns with [Name] tags
            if chunk_entities:
                # Add placeholders []
                chunk_with_placeholders = add_placeholder_regex(chunk)
                num_brackets = chunk_with_placeholders.count("[]")
                print(f"   ℹ️ Injection: {num_brackets} placeholders identified.")

                if num_brackets == 0:
                    annotated_chunk = chunk
                else:
                    # PREPEND CONTEXT to the prompt text
                    # Context has NO placeholders, so it won't affect num_brackets count logic for the chunk
                    if prev_chunk_last_sentences:
                        text_for_prompt = f"{prev_chunk_last_sentences} {chunk_with_placeholders}"
                        print(f"   ℹ️ Prepending Context: {prev_chunk_last_sentences[:50]}...")
                    else:
                        text_for_prompt = chunk_with_placeholders

                    # Build prompt
                    injection_prompt = build_entity_injection_prompt(text_for_prompt, chunk_entities, num_brackets)
                    
                    while True:
                        # Single call, no retry loop needed for consistency
                        print(f"   ℹ️ Injection Prompt Tokens: {count_tokens(injection_prompt)}")
                        response = self.call_llm_with_retry(
                            injection_prompt, 
                            const.TEMP_INJECTION, 
                            check_chinese=True,
                            llm_engine=self.injection_llm
                        )

                        # [FIX] Remove <think>...</think> blocks if they exist
                        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

                        # Check output items
                        response_items = re.findall(r"\[.*?\]", response)
                        output_count = len(response_items)
                        
                        if output_count != num_brackets:
                            print(f"\t   ⚠️ Valid Mismatch (Expected {num_brackets}, Got {output_count}). Auto-retrying...")
                            continue
                        
                        # Reconstruct text (Using ORIGINAL chunk_with_placeholders to discard context)
                        print(f"\t   ✅ Counts match. Reconstructing text...")
                        parts = chunk_with_placeholders.split("[]")
                        reconstructed = ""
                        for i in range(len(response_items)):
                            reconstructed += parts[i] + response_items[i]
                        reconstructed += parts[-1]
                        
                        annotated_chunk = reconstructed

                        # --- INTERACTIVE PAUSE FOR INJECTION ---
                        print(f"\n   --- INJECTION RESULT ---")
                        print(annotated_chunk)
                        print(f"   ------------------------")
                        user_resp = input("   [Injection] Press Enter to continue, 'r' to retry: ")
                        if user_resp.lower().strip() == 'r':
                            print("   🔄 Retrying Injection...")
                            continue
                        else:
                            break
            
            else:
                annotated_chunk = chunk  # No entities found, skip injection
            
            # Step 5: Build translation prompt with annotated text
            prompt = build_translation_prompt(
                annotated_chunk,  # Use annotated text instead of original chunk
                self.gm.glossary, 
                relations_data,
                summary, 
                model_type=const.MODEL_TYPE
            )
            
            while True:
                print(f"   ℹ️ Translation Prompt Tokens: {count_tokens(prompt)}")
                response = self.call_llm_with_retry(prompt, const.TEMP_CREATIVE)
                
                # Post-process (Extract translation only)
                pattern = r"\s*(?:[\#\*]*\s*(?:DỊCH THUẬT|BẢN DỊCH)[\#\*]*)\s*(?:[:\n]|\Z)"
                parts = re.split(pattern, response, flags=re.IGNORECASE)
                trans_text = parts[1].strip() if len(parts) > 1 else response

                # Step 6: Clean entity tags from final output
                clean_trans = clean_entity_tags(trans_text)
                
                # --- INTERACTIVE PAUSE FOR TRANSLATION ---
                print(f"\n   --- TRANSLATION RESULT ---")
                print(clean_trans)
                print(f"   --------------------------")
                user_resp = input("   [Translation] Press Enter to accept, 'r' to retry: ")
                if user_resp.lower().strip() == 'r':
                    print("   🔄 Retrying Translation...")
                    continue
                else:
                    break
            
            full_trans += clean_trans + "\n\n"

            # --- INCREMENTAL SAVE ---
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(clean_trans + "\n\n")
            
            # Step 7: Update Context for NEXT chunk (original chunk text)
            from utils import get_last_sentences
            current_last_sentences = get_last_sentences(chunk, n=2)
            prev_chunk_last_sentences = current_last_sentences

            # Update Progress
            self.gm.update_chapter_progress(chap_idx, idx, const.CHUNK_SIZE, last_sentences=current_last_sentences)
            print(f"   ✅ Chunk {idx+1} saved.")
            
        return full_trans


class PostEditingPipeline(BasePipeline):
    """
    Two-pass Draft-then-Edit pipeline for improved fluency + consistency.
    
    Phase 1: Knowledge Mining (reuses GraphRAGPipeline logic)
    Phase 2: Translation via:
        - Step A: Draft translation (fluent, no constraints)
        - Step B: Entity injection (reference generation)
        - Step C: Post-editing (fix pronouns/terminology)
    """
    def __init__(self, llm_backend, graph_manager, injection_llm=None):
        super().__init__(llm_backend, graph_manager)
        self.injection_llm = injection_llm

    def process_chapter(self, text, chap_idx, out_path):
        print(f"🚀 RUNNING: PostEditing Pipeline for Chapter {chap_idx}")
        
        # --- PHASE 1: MINING (Only if new chapter) ---
        if chap_idx > self.gm.tracking["last_processed_chap"]:
            print("   🛠️  Phase 1: Knowledge Mining...")
            self._phase_1_mining(text, chap_idx)
            self.gm.update_tracking(chap_idx)
            self.gm.save_data()
        else:
            print("   ⏭️  Phase 1 Skipped (Already processed).")

        # --- MANUAL PAUSE & RELOAD ---
        print("\n   ⏸️  Paused. You can now edit 'term_glossary.json' or 'global_graph.json'.")
        user_input = input("   Press Enter (or 'c') to continue to Phase 2, or Ctrl+C to exit: ")
        
        print("   🔄 Reloading graph data from disk...")
        self.gm.load_data()
        print("   ✅ Data reloaded.")

        # --- PHASE 2: DRAFT-THEN-EDIT TRANSLATION ---
        print("   ✍️  Phase 2: Draft-then-Edit Translation...")
        translation = self._phase_2_draft_and_edit(text, chap_idx, out_path)
        return translation

    def _phase_1_mining(self, full_text, chap_idx):
        """Reuse mining logic from GraphRAGPipeline."""
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE_MINING)
        current_glossary = {}
        
        for idx, chunk in enumerate(chunks):
            print(f"\t Đang mine chunk thứ {idx+1}/{len(chunks)} của chapter {chap_idx}")
            # 1. Mining Nodes/Edges
            candidates = filter_similar_names(get_spacy_candidates(chunk))
            existing = [n for n in self.gm.get_existing_node_ids() if n in candidates]
            
            prompt = build_extraction_prompt(chunk, existing, candidates)
            print(f"   ℹ️ Prompt Tokens: {count_tokens(prompt)}")
            
            # Retry loop for internal logic errors
            max_logic_retries = 3
            for attempt in range(max_logic_retries):
                response = self.call_llm_with_retry(prompt, const.TEMP_MINING)
                # print("----- Response -------- \n", response)
                updates = extract_json_from_response(response)
                
                if updates:
                    try:
                        self.gm.update_graph_state(updates, chap_idx, candidates)
                        break
                    except Exception as e:
                        print(f"   ⚠️ Graph Update Error: {e}. Retrying ({attempt+1}/{max_logic_retries})...")
                else:
                    if attempt < max_logic_retries - 1:
                        print(f"   ⚠️ No valid updates parsed. Retrying ({attempt+1}/{max_logic_retries})...")
                    else:
                        print("   ❌ Failed to mine updates after retries.")

            # 2. Mining Glossary
            gloss_prompt = build_glossary_extraction_prompt(chunk)
            print(f"   ℹ️ Glossary Prompt Tokens: {count_tokens(gloss_prompt)}")
            gloss_resp = self.call_llm_with_retry(gloss_prompt, 0.05, check_chinese=True)
            gloss_data = extract_json_from_response(gloss_resp)
            if gloss_data and "glossary" in gloss_data:
                current_glossary.update(gloss_data["glossary"])

        # Update global stats
        self.gm.update_node_statistics(full_text, chap_idx)
        self.gm.update_glossary(current_glossary)
        
        # Summarize
        summary_prompt = [{"role": "user", "content": f"Summarize this chapter in English (max 35 words):\n\n{full_text}"}]
        print(f"   ℹ️ Summary Prompt Tokens: {count_tokens(summary_prompt)}")
        summary = self.call_llm_with_retry(summary_prompt, const.TEMP_MINING)
        self.gm.add_history(chap_idx, summary)

    def _phase_2_draft_and_edit(self, full_text, chap_idx, out_path):
        """Two-pass translation: Draft then Edit."""
        chunks = split_text_with_overlap(full_text, const.CHUNK_SIZE, 0)
        full_trans = ""
        
        # Check Tracking to Resume
        progress = self.gm.get_chapter_progress(chap_idx)
        start_chunk_idx = 0
        
        # --- Context Management ---
        prev_chunk_last_sentences = "" 

        if progress:
            saved_size = progress.get("chunk_size", 0)
            if saved_size == const.CHUNK_SIZE:
                last_done = progress.get("last_chunk_index", -1)
                start_chunk_idx = last_done + 1
                prev_chunk_last_sentences = progress.get("last_sentences", "")
                if start_chunk_idx < len(chunks):
                    print(f"   ⏩ Resuming from chunk {start_chunk_idx+1} (Previous progress found).")
            else:
                print(f"   ⚠️ Chunk size changed ({saved_size} -> {const.CHUNK_SIZE}). Restarting chapter.")

        # If starting from scratch, clear the output file
        if start_chunk_idx == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")
            
            # Load context from previous chapter if available
            if chap_idx > 1:
                prev_chap_progress = self.gm.get_chapter_progress(chap_idx - 1)
                if prev_chap_progress:
                    prev_chunk_last_sentences = prev_chap_progress.get("last_sentences", "")
                    print(f"   🔗 Context loaded from Chapter {chap_idx - 1}")

        summary = next((h['summary'] for h in self.gm.local_graph['history'] if h['chapter'] == chap_idx), "")
        
        for idx, chunk in enumerate(chunks):
            if idx < start_chunk_idx:
                continue

            print(f"\n\t📝 Processing chunk {idx+1}/{len(chunks)}")
            
            # === STEP A: DRAFTING ===
            print(f"\t   [A] Drafting...")
            # draft_prompt = build_draft_translation_prompt(chunk, prev_chunk_last_sentences)
            draft_prompt = build_draft_translation_prompt(chunk, "")
            print(f"   ℹ️ Draft Prompt Tokens: {count_tokens(draft_prompt)}")
            draft_trans_raw = self.call_llm_with_retry(draft_prompt, const.TEMP_CREATIVE)
            
            draft_trans = draft_trans_raw
            # # Remove context overlap from draft (model may include translated context)
            # if prev_chunk_last_sentences:
            #     draft_sentences = list(nlp(draft_trans_raw).sents)
            #     if len(draft_sentences) > 2:
            #         draft_trans = " ".join([s.text for s in draft_sentences[2:]])
            #     else:
            #         draft_trans = draft_trans_raw
            # else:
            #     draft_trans = draft_trans_raw
            
            # === STEP B: ENTITY INJECTION (Reference Only) ===
            print(f"\t   [B] Generating reference...")
            chunk_entities = filter_similar_names(get_spacy_candidates(chunk))
            self.gm.update_active_memory(chunk_entities)
            
            if chunk_entities:
                chunk_with_placeholders = add_placeholder_regex(chunk)
                num_brackets = chunk_with_placeholders.count("[]")
                print(f"   ℹ️ Injection: {num_brackets} placeholders identified.")
                
                if num_brackets > 0:
                    # Prepend context for better coreference
                    if prev_chunk_last_sentences:
                        text_for_prompt = f"{prev_chunk_last_sentences} {chunk_with_placeholders}"
                    else:
                        text_for_prompt = chunk_with_placeholders
                    
                    injection_prompt = build_entity_injection_prompt(
                        text_for_prompt, chunk_entities, num_brackets
                    )
                    print(f"   ℹ️ Injection Prompt Tokens: {count_tokens(injection_prompt)}")
                    
                    response = self.call_llm_with_retry(
                        injection_prompt, const.TEMP_INJECTION,
                        llm_engine=self.injection_llm
                    )
                    
                    # Remove <think> blocks if present
                    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    
                    # Reconstruct injected text
                    response_items = re.findall(r"\[.*?\]", response)
                    if len(response_items) == num_brackets:
                        parts = chunk_with_placeholders.split("[]")
                        injected_ref = ""
                        for i, item in enumerate(response_items):
                            injected_ref += parts[i] + item
                        injected_ref += parts[-1]
                    else:
                        print(f"\t   ⚠️ Injection mismatch (Expected {num_brackets}, Got {len(response_items)}). Using original.")
                        injected_ref = chunk
                else:
                    injected_ref = chunk
            else:
                injected_ref = chunk
            
            # === STEP C: POST-EDITING ===
            print(f"\t   [C] Post-editing...")
            relevant_glossary, relevant_relations = self.gm.get_relevant_context(chunk)
            
            edit_prompt = build_post_editing_prompt(
                source_text=chunk,
                draft_text=draft_trans,
                injected_reference=injected_ref,
                glossary=relevant_glossary,
                relations=relevant_relations
            )
            print(f"   ℹ️ Edit Prompt Tokens: {count_tokens(edit_prompt)}")
            final_trans = self.call_llm_with_retry(edit_prompt, const.TEMP_EDITING)
            
            # --- INTERACTIVE REVIEW ---
            print(f"\n   --- DRAFT ---")
            print(draft_trans)
            print(f"\n   --- FINAL ---")
            print(final_trans)
            print(f"   --------------------------")
            user_resp = input("   [Translation] Press Enter to accept, 'r' to retry: ")
            if user_resp.lower().strip() == 'r':
                print("   🔄 Retrying...")
                continue
            
            # --- SAVE ---
            full_trans += final_trans + "\n\n"
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(final_trans + "\n\n")
            
            # Update context for next chunk
            current_last_sentences = get_last_sentences(chunk, n=2)
            prev_chunk_last_sentences = current_last_sentences
            
            # Update Progress
            self.gm.update_chapter_progress(chap_idx, idx, const.CHUNK_SIZE, last_sentences=current_last_sentences)
            print(f"   ✅ Chunk {idx+1} saved.")
            
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