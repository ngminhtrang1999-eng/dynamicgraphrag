import os
import re
from utils import load_json, save_json, infer_pronoun_pair

class GraphManager:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.path_global = os.path.join(data_dir, "global_graph.json")
        self.path_local = os.path.join(data_dir, "local_graph.json")
        self.path_tracking = os.path.join(data_dir, "global_tracking.json")
        self.path_glossary = os.path.join(data_dir, "term_glossary.json")
        self.path_trans_progress = os.path.join(data_dir, "translation_progress.json")

        
        # In-memory storage
        self.global_graph = {"nodes": {}, "semantic_edges": {}}
        self.local_graph = {"history": []}
        self.glossary = {}
        self.tracking = {"last_processed_chap": 0}
        
        # Incremental Translation Progress
        # Structure: {str(chap_idx): {"last_chunk_index": int, "chunk_size": int}}
        self.trans_progress = {}

        
        # Sliding Window Entity Memory
        self.active_entities = {}  # {entity_id: TTL}
        self.DEFAULT_TTL = 3  # Persists for 2 additional chunks after last mention

    def load_data(self):
        self.global_graph = load_json(self.path_global, self.global_graph)
        self.local_graph = load_json(self.path_local, self.local_graph)
        self.glossary = load_json(self.path_glossary, self.glossary)
        self.tracking = load_json(self.path_tracking, self.tracking)
        self.trans_progress = load_json(self.path_trans_progress, self.trans_progress)

        # A1 schema migration: flat {term: "translation"} -> {term: {"translation": "...", "type": "UNKNOWN"}}.
        # Legacy entries become typed-on-read so downstream code can assume the structured form.
        migrated = 0
        for term, value in list(self.glossary.items()):
            if isinstance(value, str):
                self.glossary[term] = {"translation": value, "type": "UNKNOWN"}
                migrated += 1
            elif isinstance(value, dict):
                value.setdefault("translation", "")
                value.setdefault("type", "UNKNOWN")
        if migrated:
            print(f"      [A1] migrated {migrated} flat glossary entries to typed schema")


    def save_data(self):
        save_json(self.path_global, self.global_graph)
        save_json(self.path_local, self.local_graph)
        save_json(self.path_glossary, self.glossary)
        save_json(self.path_tracking, self.tracking)
        save_json(self.path_trans_progress, self.trans_progress)


    def get_existing_node_ids(self):
        return list(self.global_graph["nodes"].keys())

    def update_tracking(self, chap_idx):
        self.tracking["last_processed_chap"] = chap_idx

    def add_history(self, chap_idx, summary):
        self.local_graph["history"].append({"chapter": chap_idx, "summary": summary})

    def update_glossary(self, new_terms):
        """Merge new extraction output into the glossary.

        Accepts either the new typed schema (`{term: {translation, type}}`) or the
        legacy flat schema (`{term: "translation"}`). String values are coerced
        into typed entries with type=UNKNOWN so the on-disk format stays uniform.
        """
        if not new_terms:
            return
        for term, value in new_terms.items():
            if isinstance(value, dict):
                entry = {
                    "translation": value.get("translation", "").strip(),
                    "type": (value.get("type") or "UNKNOWN").upper(),
                }
            else:
                entry = {"translation": str(value).strip(), "type": "UNKNOWN"}
            if not entry["translation"]:
                continue
            # Preserve type if a previously typed entry is being updated with an UNKNOWN one.
            existing = self.glossary.get(term)
            if isinstance(existing, dict) and entry["type"] == "UNKNOWN" and existing.get("type", "UNKNOWN") != "UNKNOWN":
                entry["type"] = existing["type"]
            self.glossary[term] = entry

    # --- PROGRESS TRACKING (TRANSLATION) ---
    def get_chapter_progress(self, chap_idx):
        prog = self.trans_progress.get(str(chap_idx), {}).copy()
        # Convert 1-based (human friendly) back to 0-based for internal logic
        if "last_chunk_index" in prog:
            prog["last_chunk_index"] -= 1
        return prog

    def update_chapter_progress(self, chap_idx, chunk_index, chunk_size, last_sentences=""):
        self.trans_progress[str(chap_idx)] = {
            "last_chunk_index": chunk_index + 1, # Make it 1-based for human readability
            "chunk_size": chunk_size,
            "last_sentences": last_sentences
        }
        # Save immediately to ensure resume capability
        save_json(self.path_trans_progress, self.trans_progress)

    # --- SLIDING WINDOW ENTITY MEMORY ---
    def _resolve_to_graph_id(self, raw_name):
        """Resolve a raw entity name to its canonical graph ID."""
        if not raw_name:
            return None
        if raw_name in self.global_graph["nodes"]:
            return raw_name
        raw_lower = raw_name.lower()
        for node_id in self.global_graph["nodes"]:
            if raw_lower in node_id.lower() or node_id.lower() in raw_lower:
                return node_id
        return None

    def update_active_memory(self, entities_found):
        """Update sliding window memory with entities from current chunk.
        
        Args:
            entities_found: List of entity names found in current chunk (from NER)
        """
        # Step 1: Reset TTL for found entities
        for entity in entities_found:
            canonical_id = self._resolve_to_graph_id(entity)
            if canonical_id:
                self.active_entities[canonical_id] = self.DEFAULT_TTL
        
        # Step 2: Decay all TTLs
        expired = []
        for entity_id in self.active_entities:
            self.active_entities[entity_id] -= 1
            if self.active_entities[entity_id] <= 0:
                expired.append(entity_id)
        
        # Step 3: Clean expired entities
        for entity_id in expired:
            del self.active_entities[entity_id]

    def get_relations_data(self):
        """Get relations for active entities as structured data.
        
        Returns:
            List[Dict]: Each dict contains {source, target, source_pronoun, 
                        target_pronoun, relation_type}
        """
        relations = []
        nodes_map = self.global_graph.get("nodes", {})
        active_set = set(self.active_entities.keys())
        
        for key, data in self.global_graph["semantic_edges"].items():
            src = data["source"]
            tgt = data["target"]
            
            # Check if at least one entity is active
            if src in active_set or tgt in active_set:
                src_node = nodes_map.get(src)
                tgt_node = nodes_map.get(tgt)
                
                # Only include PERSON-PERSON relationships
                if src_node and tgt_node:
                    
                    curr = data["current"]
                    pronouns = curr.get("pronouns", {})
                    
                    s_call_t = pronouns.get("source_call_target", "")
                    t_call_s = pronouns.get("target_call_source", "")

                    # Infer "Self" pronouns (Ta/Tôi/Con/...) based on how they call the other
                    # Also use gender (if available) to refine the self-pronoun (e.g. Chú/Cô vs Cháu)
                    src_gender = src_node.get("gender", "UNKNOWN")
                    tgt_gender = tgt_node.get("gender", "UNKNOWN")
                    
                    s_call_self = infer_pronoun_pair(s_call_t, gender=src_gender)
                    t_call_self = infer_pronoun_pair(t_call_s, gender=tgt_gender)
                    
                    relations.append({
                        "source": src,
                        "target": tgt,
                        "s_call_t": s_call_t,
                        "t_call_s": t_call_s,
                        "s_call_self": s_call_self,
                        "t_call_self": t_call_self, # Adding this for completeness if needed (target's perspective)
                        "relation_type": curr.get("relation", "related")
                    })
        
        return relations

    def get_relevant_context(self, chunk_text):
        """
        Filter glossary and relations to only include items relevant to current chunk.
        
        Args:
            chunk_text: The current chunk text
            
        Returns:
            Tuple[dict, list]: (filtered_glossary, filtered_relations)
            
        Note:
            Simple substring matching may cause false positives (e.g., "ear" matches "bear").
            Acceptable for thesis scope; consider regex \bterm\b for stricter matching if needed.
        """
        chunk_lower = chunk_text.lower()

        # Filter glossary: only terms that appear in chunk (simple substring match).
        # Output is a flat {term: entry_dict} mapping; type-bucketing happens in
        # the prompt-rendering layer so callers that just want lookup keep working.
        filtered_glossary = {}
        for term, entry in self.glossary.items():
            if term.lower() not in chunk_lower:
                continue
            if isinstance(entry, dict):
                filtered_glossary[term] = {
                    "translation": entry.get("translation", ""),
                    "type": (entry.get("type") or "UNKNOWN").upper(),
                }
            else:
                # Legacy in-memory state (shouldn't happen after load_data migration, but safe)
                filtered_glossary[term] = {"translation": str(entry), "type": "UNKNOWN"}
        
        # Filter relations: Get from active memory first
        all_relations = self.get_relations_data()
        
        # Stopwords to skip when matching first name
        stopwords = {'the', 'a', 'an', 'of', 'and', 'or'}
        
        # Helper: Check if entity name appears in text (handles "Klein" matching "Klein Moretti")
        def entity_in_text(entity_name, text_lower):
            if not entity_name:
                return False
            # Check full name
            if entity_name.lower() in text_lower:
                return True
            # Get significant words (skip stopwords)
            words = [w for w in entity_name.split() if w.lower() not in stopwords]
            if not words:
                return False
            # Check first significant word (e.g., "The Fool" -> check "Fool")
            first_significant = words[0]
            if first_significant.lower() in text_lower:
                return True
            return False
        
        # Filter: only keep relations where BOTH entities appear in chunk
        filtered_relations = []
        for r in all_relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            # Require BOTH source and target to appear in chunk
            if entity_in_text(src, chunk_lower) and entity_in_text(tgt, chunk_lower):
                filtered_relations.append(r)
        
        return filtered_glossary, filtered_relations

    # --- CORE GRAPH LOGIC (Moved from test.py) ---
    def resolve_entity_name(raw_name, existing_nodes):
        """Logic hợp nhất tên (Entity Resolution)"""
        if not raw_name: return ""
        raw_clean = raw_name.lower().strip()
        if raw_name in existing_nodes: return raw_name
        for node_id in existing_nodes:
            node_clean = node_id.lower()
            if (raw_clean == node_clean) or \
            (raw_clean in node_clean.split() and len(node_id) > len(raw_name)): 
                return node_id          
        return raw_name

    def update_node_statistics(self, text, chapter_id):
        nodes = self.global_graph["nodes"]
        nodes_to_remove = []
        
        # Danh sách các tiền tố không nên tính là tên gọi tắt (tránh nhầm lẫn)
        # Ví dụ: Node là "Detective Sherlock", ta không muốn tách ra alias "Detective" để đếm
        ignore_prefixes = {'mr', 'mr.', 'ms', 'ms.', 'dr', 'dr.', 'miss', 'mrs', 'mrs.', 'detective', 'captain'}
        
        for node_id, node_data in nodes.items():
            # 1. Tự động sinh các biến thể tên (Aliases)
            variants = {node_id} # Luôn bắt đầu bằng tên gốc (Full ID)
            
            parts = node_id.split()
            if len(parts) > 1:
                # Lấy tên đầu (First Name) - Phổ biến nhất
                first_part = parts[0]
                # Chỉ lấy nếu nó không phải danh xưng (Mr, Dr...) và đủ dài (>2 ký tự)
                if first_part.lower() not in ignore_prefixes and len(first_part) > 2:
                    variants.add(first_part)
                
                # (Tùy chọn) Lấy tên cuối (Last Name)
                # last_part = parts[-1]
                # if last_part.lower() not in ignore_prefixes and len(last_part) > 2:
                #     variants.add(last_part)

            # 2. Sắp xếp biến thể theo độ dài giảm dần (QUAN TRỌNG)
            # Để Regex khớp chuỗi dài nhất trước: "Klein Moretti" ưu tiên hơn "Klein"
            sorted_variants = sorted(list(variants), key=len, reverse=True)
            
            # 3. Tạo Regex tổng hợp: \b(Klein Moretti|Klein)\b
            # Dùng re.escape để xử lý các ký tự đặc biệt nếu có
            pattern_str = r"\b(?:" + "|".join(re.escape(v) for v in sorted_variants) + r")\b"
            
            # 4. Đếm số lần xuất hiện
            matches = re.findall(pattern_str, text, re.IGNORECASE)
            count_in_chapter = len(matches)
            
            if count_in_chapter > 0:
                # Cộng dồn số lần xuất hiện thực tế
                node_data["mention_count"] = node_data.get("mention_count", 0) + count_in_chapter
                node_data["last_seen"] = chapter_id
                
                # Logic nâng cấp lên Permanent
                if node_data.get("status_graph", "Probation") == "Probation" and node_data["mention_count"] >= 10:
                    node_data["status_graph"] = "Permanent"
                    print(f"      ★ [UPGRADE] Node Permanent: {node_id} (Count: {node_data['mention_count']})")
    
        # Garbage Collection (Logic cũ)
        for node_id, node_data in nodes.items():
            if node_data.get("status_graph", "Probation") == "Probation":
                if (chapter_id - node_data.get("last_seen", chapter_id)) >= 15: 
                    nodes_to_remove.append(node_id)

        if nodes_to_remove:
            print(f"      🗑️ [CLEANUP] Removing {len(nodes_to_remove)} probation nodes")
            for nid in nodes_to_remove: 
                del self.global_graph["nodes"][nid]
                
        return self.global_graph
    
    def update_graph_state(self, updates, chapter_id, candidates):
        """
        Update logic with Entity Resolution & Data Consolidation
        Tự động phát hiện tên dài/ngắn để gộp node và cập nhật quan hệ (Nodes + Edges).
        """
        if not updates: return self.global_graph
        
        # --- FIX: Handle List Input (LLM fallback) ---
        if isinstance(updates, list):
            # Case 1: List containing the data dict -> [{"nodes": ...}]
            if len(updates) > 0 and isinstance(updates[0], dict) and "nodes" in updates[0]:
                updates = updates[0]
            # Case 2: List of nodes directly -> [{"id": "Klein", ...}, {"id": "Dunn", ...}]
            elif all(isinstance(x, dict) and "id" in x for x in updates):
                updates = {"nodes": updates, "edges": []}
            else:
                 # print(f"      ⚠️ Warning: `updates` is a list with unknown format. Using empty dict.")
                 # updates = {}
                 raise ValueError("Invalid JSON format: List with unknown structure")

        candidates_lower = [c.lower() for c in candidates]
        
        if "semantic_edges" not in self.global_graph: self.global_graph["semantic_edges"] = {}
        
        # Danh sách từ loại bỏ khi so sánh tên
        stop_words = {'mr', 'mr.', 'ms', 'ms.', 'dr', 'dr.', 'miss', 'mrs', 'mrs.'}
        
        # Map để lưu chuyển đổi ID tạm thời trong lần update này (Input ID -> Graph ID)
        resolved_id_map = {} 
        
        # ================= 1. NODES PROCESSING (SMART MERGE) =================
        for node in updates.get("nodes", []): 
            raw_id = node.get("id")
            if not raw_id: continue
            
            # Chỉ xử lý nếu node nằm trong danh sách candidates (đã filter từ NER)
            # hoặc nếu bạn muốn tin tưởng hoàn toàn LLM thì bỏ dòng này
            if raw_id.lower() not in candidates_lower: continue

            # Tạo tokens cho ID mới từ Update
            raw_tokens = set(raw_id.lower().split())
            raw_tokens = {t for t in raw_tokens if t not in stop_words}
            
            target_id = raw_id # Mặc định là chính nó
            match_found = False

            # --- LOGIC TÌM KIẾM & GỘP TRONG GLOBAL GRAPH ---
            existing_ids = list(self.global_graph["nodes"].keys())
            
            for exist_id in existing_ids:
                exist_tokens = set(exist_id.lower().split())
                exist_tokens = {t for t in exist_tokens if t not in stop_words}
                
                # Case A: Trùng khớp hoàn toàn hoặc trùng ID
                if raw_id == exist_id:
                    target_id = exist_id
                    match_found = True
                    break
                    
                # Case B: Input là con của Existing (VD: Input="Klein" -> Existing="Klein Moretti")
                if raw_tokens and raw_tokens.issubset(exist_tokens) and len(raw_id) < len(exist_id):
                    target_id = exist_id
                    match_found = True
                    break
                
                # Case C: Existing là con của Input (VD: Input="Klein Moretti" -> Existing="Klein")
                # => NÂNG CẤP Existing thành Input
                if exist_tokens and exist_tokens.issubset(raw_tokens) and len(raw_id) > len(exist_id):
                    print(f"      🔄 [MERGE] Upgrading Node: '{exist_id}' -> '{raw_id}'")
                    
                    old_data = self.global_graph["nodes"].pop(exist_id)
                    old_data["id"] = raw_id
                    if node.get("gender") and node.get("gender") != "UNKNOWN":
                        old_data["gender"] = node.get("gender")
                    
                    self.global_graph["nodes"][raw_id] = old_data
                    
                    # Lưu vào map để tí nữa update edge cũ
                    resolved_id_map[exist_id] = raw_id # Old -> New
                    
                    target_id = raw_id
                    match_found = True
                    break
            
            # Lưu map cho node hiện tại (Input -> Final Graph ID)
            resolved_id_map[raw_id] = target_id

            # --- CẬP NHẬT STATS CHO TARGET NODE ---
            if target_id not in self.global_graph["nodes"]:
                # Node mới
                self.global_graph["nodes"][target_id] = {
                    "id": target_id,
                    "gender": node.get("gender", "UNKNOWN"),
                    "status_graph": "Probation",
                    "mention_count": 0,
                    "first_seen": chapter_id,
                    "last_seen": chapter_id
                }
            else:
                # Node cũ
                curr_node = self.global_graph["nodes"][target_id]
                curr_node["last_seen"] = chapter_id

        # ================= 1.5 FIX OLD EDGES IF KEY CHANGED (Case C) =================
        if any(k != v for k, v in resolved_id_map.items()):
            new_edges_dict = {}
            for key, edge_data in self.global_graph["semantic_edges"].items():
                src = edge_data["source"]
                tgt = edge_data["target"]
                
                # Cập nhật ID nếu node bị đổi tên (Case C)
                new_src = resolved_id_map.get(src, src)
                new_tgt = resolved_id_map.get(tgt, tgt)
                
                if new_src == new_tgt: continue 

                edge_data["source"] = new_src
                edge_data["target"] = new_tgt
                
                new_key = "|".join(sorted([new_src, new_tgt]))
                
                if new_key in new_edges_dict:
                    # Merge history nếu 2 edge cũ gộp thành 1
                    existing_edge = new_edges_dict[new_key]
                    existing_edge["history"].extend(edge_data.get("history", []))
                    if edge_data["current"]["since"] > existing_edge["current"]["since"]:
                        existing_edge["current"] = edge_data["current"]
                else:
                    new_edges_dict[new_key] = edge_data
                    
            self.global_graph["semantic_edges"] = new_edges_dict

# ================= 2. EDGES PROCESSING (NEW DATA) =================
        if "edges" in updates:
            for edge in updates["edges"]:
                src_raw = edge.get("source")
                tgt_raw = edge.get("target")

                if not src_raw or not tgt_raw: continue

                # Map ID input sang ID chuẩn
                src = resolved_id_map.get(src_raw, src_raw)
                tgt = resolved_id_map.get(tgt_raw, tgt_raw)

                if src == tgt: continue

                if src in self.global_graph["nodes"] and tgt in self.global_graph["nodes"]:
                    
                    edge_key = "|".join(sorted([src, tgt]))

                    # 1. Lấy dữ liệu Pronouns MỚI từ update
                    pronouns_data = edge.get("pronouns") or {}
                    new_src_call_tgt = pronouns_data.get("source_call_target", "")
                    new_tgt_call_src = pronouns_data.get("target_call_source", "")
                    
                    # Tạo object thông tin quan hệ mới
                    relation_info = {
                        "relation": edge.get("relation_type", "related"),
                        "pronouns": {
                            "source_call_target": new_src_call_tgt,
                            "target_call_source": new_tgt_call_src
                        },
                        "since": chapter_id
                    }

                    if edge_key not in self.global_graph["semantic_edges"]:
                        # --- TẠO EDGE MỚI (CHƯA TỒN TẠI) ---
                        self.global_graph["semantic_edges"][edge_key] = {
                            "source": src,
                            "target": tgt,
                            "history": [],
                            "current": relation_info
                        }
                    else:
                        # --- XỬ LÝ EDGE ĐÃ TỒN TẠI ---
                        existing_edge = self.global_graph["semantic_edges"][edge_key]
                        current_data = existing_edge["current"]

                        # =========================================================
                        # LOGIC 1: CÙNG CHAPTER -> GỘP THÔNG TIN (MERGE)
                        # =========================================================
                        if current_data["since"] == chapter_id:
                            # 1. Cập nhật Relation (Lấy cái mới nhất vì thường nó cụ thể hơn)
                            current_data["relation"] = relation_info["relation"]

                            # 3. Cập nhật Pronouns (Chỉ cập nhật nếu MỚI có dữ liệu, KHÔNG ghi đè bằng rỗng)
                            new_s_call_t = relation_info["pronouns"]["source_call_target"]
                            new_t_call_s = relation_info["pronouns"]["target_call_source"]

                            if new_s_call_t:
                                current_data["pronouns"]["source_call_target"] = new_s_call_t
                            
                            if new_t_call_s:
                                current_data["pronouns"]["target_call_source"] = new_t_call_s
                        
                        # =========================================================
                        # LOGIC 2: KHÁC CHAPTER -> ĐẨY VÀO HISTORY (ARCHIVE)
                        # =========================================================
                        else:
                            # Lấy pronouns cũ để kế thừa
                            old_pronouns = current_data.get("pronouns", {})

                            # Logic kế thừa xưng hô (như đã bàn ở câu trước)
                            # Nếu chương mới không có xưng hô -> Lấy lại của chương cũ
                            if not relation_info["pronouns"]["source_call_target"]:
                                relation_info["pronouns"]["source_call_target"] = old_pronouns.get("source_call_target", "")
                            
                            if not relation_info["pronouns"]["target_call_source"]:
                                relation_info["pronouns"]["target_call_source"] = old_pronouns.get("target_call_source", "")

                            # Đẩy trạng thái cũ vào history
                            existing_edge["history"].append(current_data)
                            
                            # Cập nhật trạng thái hiện tại sang Chapter mới
                            existing_edge["current"] = relation_info
                            
                            # Cập nhật lại source/target (đề phòng đổi chiều quan hệ ở chap mới)
                            existing_edge["source"] = src
                            existing_edge["target"] = tgt

        return self.global_graph
    
    def get_relations_string(self, text_chunk):
        rels = []
        # Lấy danh sách nodes để check type
        nodes_map = self.global_graph.get("nodes", {}) 
        
        for key, data in self.global_graph["semantic_edges"].items():
            src = data["source"]
            tgt = data["target"]
            
            # 1. Chỉ xử lý nếu ít nhất 1 trong 2 thực thể xuất hiện trong text
            if src in text_chunk or tgt in text_chunk:
                
                # Lấy data của node source và target
                src_node = nodes_map.get(src)
                tgt_node = nodes_map.get(tgt)
                
                # 2. CORE LOGIC: Chỉ lấy nếu cả 2 đều tồn tại
                if src_node and tgt_node:
                    
                    curr = data["current"]
                    p = curr.get("pronouns", {})
                    
                    # # Xử lý fallback nếu model trả về string rỗng ""
                    s_call_t = p.get("source_call_target", "")
                    # if not s_call_t: s_call_t = "Ta" # Default
                    
                    t_call_s = p.get("target_call_source", "") 
                    # if not t_call_s: t_call_s = "Ngươi" # Default

                    rels.append(f"- {src} calls {tgt}: '{s_call_t}' | {tgt} calls {src}: '{t_call_s}' ({curr.get('relation', 'related')})")
                    
        return "\n".join(rels)