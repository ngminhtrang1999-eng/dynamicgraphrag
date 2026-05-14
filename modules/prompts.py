import json

# --- Helper: Linearize Constraints for Weak LLMs ---
def linearize_constraints(relations_data):
    """Convert relation data to imperative Vietnamese instructions.
    
    Args:
        relations_data: List of dicts with {source, target, source_pronoun, 
                        target_pronoun, relation_type}
    
    Returns:
        str: Linearized constraint instructions in Vietnamese
    """
    if not relations_data:
        return ""
    
    lines = ["### BẮT BUỘC TUÂN THỦ CÁC QUY TẮC XƯNG HÔ SAU:"]
    
    for rel in relations_data:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        src_pron = rel.get("source_pronoun", "")
        tgt_pron = rel.get("target_pronoun", "")
        
        if src_pron:
            lines.append(f"- Nếu {src} nói với {tgt} -> Bắt buộc xưng là \"{src_pron}\".")
        if tgt_pron:
            lines.append(f"- Nếu {tgt} nói với {src} -> Bắt buộc gọi là \"{tgt_pron}\".")
    
    return "\n".join(lines) if len(lines) > 1 else ""

# --- Helper: Build Entity Injection Prompt ---
def build_entity_injection_prompt(chunk_text, candidates, num_brackets):
    """
    Build prompt for Pre-Translation Entity Injection.
    This annotates pronouns with character names to improve coreference resolution.
    
    Args:
        chunk_text: The text to annotate
        candidates: List of character names from NER
        num_brackets: Number of placeholders in the text
    
    Returns:
        list: Messages for LLM call
    """
    candidates_list_string = ", ".join(candidates) if candidates else "No candidates available"
    
    # 1. CẬP NHẬT SYSTEM INSTRUCTION
    # Thêm quy tắc số 2 (Never delete) và nhấn mạnh lại format.
    system_instruction = """You are a Coreference Resolution Expert.
The user provides text where pronouns are followed by empty brackets `[]`.
Your task is to FILL these brackets with the correct Referent Name from the "Candidates List".

### RULES:
1. INPUT: `He [] walked...` -> OUTPUT: `[Klein]`
2. ONLY use names provided in the "Candidates List".
3. If the referent is ambiguous or missing from the list, KEEP the brackets empty: `[]`.
4. Number of output items must match the number of empty brackets in the input."""

    # 2. Few-shot Examples
    # QUAN TRỌNG: Input mẫu PHẢI CÓ SẴN [] giống như dữ liệu thật
    few_shots = [
        {
            "user": "There are 3 empty brackets. Candidates List: ['Klein Moretti', 'Dunn', 'Daly']\nInput: Klein walked into the room. He [] saw Dunn. He [] nodded at him [].",
            "assistant": "1. [Klein Moretti] 2. [Klein Moretti] 3. [Dunn]"
        },
        {
            "user": "There are 2 empty brackets. Candidates List: ['Zhou Mingrui', 'Melissa']\nInput: While feeling shocked, he [] discovered the notebook. Subconsciously, he [] looked up.",
            "assistant": "1. [Zhou Mingrui] 2. [Zhou Mingrui]"
        },
        {
            "user": "There are 2 empty brackets. Candidates List: ['Dunn Sminth', 'Klein Moretti']\nInput: As Dunn approach the man in white, Klein also approach him []. He [] also pull out a gun.",
            "assistant": "1. [] 2. [Klein Moretti]"
        }
    ]

    # 3. Xây dựng cấu trúc Messages hoàn chỉnh
    messages = [
        {"role": "system", "content": system_instruction}
    ]

    # Nạp Few-shot vào history
    for shot in few_shots:
        messages.append({"role": "user", "content": shot["user"]})
        messages.append({"role": "assistant", "content": shot["assistant"]})

    # 4. Real Request (Dữ liệu thực tế cần xử lý)
    real_user_content = f"""There are {num_brackets} empty brackets:
    Candidates List: {candidates_list_string}
    Input: {chunk_text}"""

    messages.append({"role": "user", "content": real_user_content})

    return messages

# --- Helper: Build prompt ---
def build_extraction_prompt(chunk_text, existing_nodes_list, spacy_candidates_list):
    """
    Update: Tăng cường ràng buộc Tiếng Việt và Hán Việt.
    """
    system_instruction = """Bạn là một Chuyên gia Kỹ thuật Knowledge Graph cho hệ thống dịch thuật tiểu thuyết.
Nhiệm vụ: Trích xuất Thực thể (Nodes) và Mối quan hệ (Edges) quan trọng từ văn bản.

### 1. QUY TẮC BỘ LỌC:
**CHỈ** quan tâm đến các thực thể có vai trò cốt truyện.

* **Valid Types (CHỈ CHẤP NHẬN LOẠI SAU):**
    1.  `PERSON`: Nhân vật có tên riêng hoặc danh xưng cụ thể (VD: Klein, Captain, Melissa).
    2.  `DEITY`: Chỉ các vị thần huyền bí có **Tên Riêng** (VD: The Fool, Evernight Goddess, God of Steam and Machinery).

### 2. QUY TẮC NGÔN NGỮ OUTPUT - BẮT BUỘC TUÂN THỦ:
* `reasoning`: PHẢI viết bằng **TIẾNG VIỆT**.

### 3. NODE RESOLUTION:
* Ưu tiên dùng ID từ `EXISTING_NODES` nếu trùng khớp.
* Chỉ chọn các thực thể trong danh sách `CANDIDATES`.

### 4. QUY TẮC CHỌN SOURCE VÀ TARGET:
* **CHỈ CHỌN TỪ CANDIDATES**: Chỉ trích xuất các thực thể trong danh sách `CANDIDATES` được cung cấp.

### 5. QUY TẮC SUY LUẬN MỐI QUAN HỆ (RELATION TYPE):
* Không giới hạn danh sách từ khóa. Hãy tự suy luận nhãn (Label) dựa trên **bản chất tương tác**, **cảm xúc** và **địa vị xã hội** trong đoạn văn bản.
* Định dạng: Sử dụng từ khóa tiếng Anh viết hoa (UPPERCASE).
* Ví dụ ngữ cảnh: Cùng làm việc -> `COLLEAGUE`; Đối đầu -> `HOSTILE`; Mới gặp -> `STRANGER`.

### 6. QUY TẮC XƯNG HÔ (PRONOUNS) - CỰC KỲ QUAN TRỌNG:
* **Ưu tiên 1 - Dựa trên Hội thoại**: Nếu hội thoại đủ thông tin, trích xuất cách xưng hô thực tế.
* **Ưu tiên 2 -Dựa trên Vị thế (Implicit Status)**: Nếu KHÔNG có hội thoại hoặc không đủ thông tin, suy luận từ ngữ cảnh và vai trò:
    * **DEITY / LEADER vs MEMBER**: 
        * Member -> Leader: "Ngài" (Lord/Sir), "Đội trưởng" (Captain).
        * Leader -> Member: "Ngươi" (You), "Cậu/Cô" (You).
    * **STRANGER vs STRANGER**: "Anh/Cô" - "Tôi".
    * **ENEMY**: "Hắn/Gã" - "Ta/Ngươi".
    * **Lưu ý đặc biệt**: Với thực thể là "The Fool" (Kẻ Khờ) hoặc các vị thần, Target (kẻ dưới) LUÔN LUÔN gọi là "Ngài" (My Lord), bất kể mối quan hệ.
* **Ưu tiên 3 - Fallback an toàn (QUAN TRỌNG)**: Nếu hoàn toàn không có tương tác, không có hội thoại, và không đủ dữ kiện ngữ cảnh để xác định vị thế xã hội, BẮT BUỘC điền là "UNKNOWN".

### OUTPUT SCHEMA (Strict JSON):
```json
{
  "reasoning": "1. Trích xuất 'Zhou Mingrui' vì là nhân vật chính...",
  "nodes": [
    {
      "id": "Standard_English_ID",
      "gender": "MALE" | "FEMALE" | "UNKNOWN"
    }
  ],
  "edges": [
    {
      "source": "ID_A",
      "target": "ID_B",
      "relation_type": "STRING_UPPERCASE_DESCRIBING_INTERACTION",
      "pronouns": {
        "source_call_target": "A gọi B là...",
        "target_call_source": "B gọi A là..."  
      }
    }
  ]
}
"""
    # Few-shot 1
    shot_1_user = f"""
    EXISTING_NODES: ["Klein Moretti"]
    CANDIDATES: ["Dunn Smith", "Old Neil"]
    TEXT:
    Đội trưởng Dunn Smith nhìn Klein nghiêm nghị. "Cậu đến muộn," Dunn nói. Klein cúi đầu: "Xin lỗi đội trưởng."
    """ 
    
    
    shot_1_assistant = """```json { "reasoning": "1. 'Klein' khớp với node cũ 'Klein Moretti'. 2. 'Dunn Smith' là nhân vật mới, vai trò đội trưởng (cấp trên). 3. Bối cảnh: Cấp trên khiển trách cấp dưới. 4. Xưng hô: Dunn gọi Klein là 'Cậu', Klein gọi Dunn là 'Đội trưởng'.", "nodes": [ { "id": "Klein Moretti", "gender": "MALE" }, { "id": "Dunn Smith", "gender": "MALE" } ], "edges": [ { "source": "Dunn Smith", "target": "Klein Moretti", "relation_type": "ALLY", "pronouns": { "source_call_target": "Cậu", "target_call_source": "Đội trưởng" } } ] }"""

    real_user_content = f"""
### EXISTING_NODES: {json.dumps(existing_nodes_list)}
### CANDIDATES: {json.dumps(spacy_candidates_list)}
### TEXT:
{chunk_text}
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": shot_1_user},
        {"role": "assistant", "content": shot_1_assistant},
        {"role": "user", "content": real_user_content}
    ]
    return messages


def build_glossary_extraction_prompt(chunk_text):
    system_instruction = """
Bạn là một Biên tập viên Từ điển Cấp cao cho dự án Dịch thuật Tiểu thuyết Giả tưởng (theo phong cách Quỷ Bí Chi Chủ / Lord of the Mysteries).
Nhiệm vụ của bạn là trích xuất các thuật ngữ cụ thể và cung cấp bản dịch tiếng Việt chuẩn xác nhất.

### 1. CÁC DANH MỤC MỤC TIÊU (Cần trích xuất):
* **TỔ CHỨC (ORGANIZATION)**: Các Giáo phái, Giáo hội, Biệt đội, Công ty, Hiệp hội Kinh tế. (Ví dụ: Kẻ Gác Đêm, Hiệp hội Than Thép).
* **THẦN LINH (DEITY)**: Các vị Thần, Thực thể thần thoại. (Ví dụ: Nữ Thần Đêm Đen).
* **ĐỊA DANH (LOCATION)**: Vương quốc, Thành phố, Tên đường, Các tòa nhà có tên riêng.
* **KHÁI NIỆM / DANH XƯNG / VẬT PHẨM**: 
    * **Danh sách (Pathways/Sequences)**: (Ví dụ: Nhà Tiên Tri, Khán Giả, Người Phi Phàm).
    * **Thuật ngữ Ma thuật**: (Ví dụ: Ma dược, Nghi thức, Minh tưởng).
    * **Vật phẩm cụ thể**: (Ví dụ: Bài Tarot, Vật Phong Ấn, 0-08).
    * **Danh xưng/Tiếng lóng**: (Ví dụ: Anh hùng bàn phím, Ẩn Chính).

### 2. QUY TẮC LOẠI TRỪ (Bỏ qua những từ này):
* **BỎ QUA DANH TỪ CHUNG**: Bàn, ghế, đèn, súng thường, tiền, bánh mì (trừ khi đó là một vật phẩm có tên riêng cụ thể).
* **BỎ QUA ĐỘNG TỪ/TÍNH TỪ**.

### 3. HƯỚNG DẪN DỊCH THUẬT:
* **Dịch theo ngữ cảnh**: KHÔNG dịch từng từ (word-for-word). Hãy sử dụng kiến thức chuyên ngành (domain knowledge).
    * *Ví dụ*: "Major Arcana" -> "Ẩn Chính" (Thuật ngữ Tarot), KHÔNG dịch là "Đại Arcana".
    * *Ví dụ*: "Amalgamators" (trong kinh doanh) -> "Hiệp hội/Tổ hợp", KHÔNG dịch là "Hợp kim".
* **Văn phong**: Ưu tiên sử dụng từ **Hán Việt** cho các thuật ngữ Ma thuật/Tu chân/Tôn giáo để tạo sắc thái trang trọng, cổ kính.

### ĐỊNH DẠNG ĐẦU RA (CHỈ JSON):
Mỗi thuật ngữ phải gắn nhãn `type` thuộc một trong các giá trị:
`ORG` (tổ chức), `DEITY` (thần linh), `LOCATION` (địa danh), `CONCEPT` (khái niệm / nghi lễ / pathway), `ITEM` (vật phẩm cụ thể), `TITLE` (danh xưng / tiếng lóng / chức danh).

```json
{
  "reasoning": "Giải thích lý do tại sao trích xuất các thuật ngữ này và cách chọn từ ngữ dịch thuật...",
  "glossary": {
      "English Term": { "translation": "Vietnamese Translation", "type": "CONCEPT" }
  }
}
"""
    # ... (Giữ nguyên phần còn lại của hàm, chỉ cần update system_instruction)
    # Tuy nhiên, nên update Few-shot 1 chút để model hiểu category mới
    shot_1_user = """
    TEXT:
    Klein checked the Major Arcana cards. He acted like a Keyboard Warrior in his past life.
    """
    shot_1_assistant = """```json
{
    "reasoning": "1. 'Major Arcana' is a Tarot Concept -> CONCEPT. Translate: 'Ẩn Chính'. 2. 'Keyboard Warrior' is a Slang/Title -> TITLE. Translate: 'Anh hùng bàn phím' or 'Thánh bàn phím'.",
    "glossary": {
        "Major Arcana": { "translation": "Ẩn Chính", "type": "CONCEPT" },
        "Keyboard Warrior": { "translation": "Anh hùng bàn phím", "type": "TITLE" }
    }
}"""

    # Few-shot 2: Filtering noise (Person & Common items)
    shot_2_user = """
    TEXT:
    Klein sat on the sofa in the living room. He looked at Old Neil. The Holy Cathedral was far away.
    """

    shot_2_assistant = """```json
{
  "reasoning": "1. 'Klein' and 'Old Neil' are PERSONS -> IGNORE. 2. 'sofa', 'living room' are generic objects/places -> IGNORE. 3. 'Holy Cathedral' is a named building/organization seat -> LOCATION.",
  "glossary": {
      "Holy Cathedral": { "translation": "Thánh Đường", "type": "LOCATION" }
  }
}
```"""

    real_user_content = f"""
    TEXT:
    {chunk_text}
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": shot_1_user},
        {"role": "assistant", "content": shot_1_assistant},
        {"role": "user", "content": shot_2_user},
        {"role": "assistant", "content": shot_2_assistant},
        {"role": "user", "content": real_user_content}
    ]
    return messages

def _format_relations_to_table(relations_data):
    """
    Chuyển list quan hệ thành bảng Markdown để model dễ tra cứu.
    """
    if not relations_data:
        return ""
    
    # Header bảng
    table = [
        "| Speaker (Người nói) | Listener (Người nghe) | Call_Listener (Gọi người nghe) | Call_Speaker (Gọi người nói) | Self_Speaker (Người nói xưng hô) | Self_Listener (Người gọi xưng hô)",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in relations_data:
        # Giả sử structure r = {'source': 'A', 'target': 'B', 't_call_s': 'Cậu', 's_call_t': 'Tôi', 's_call_self': 'Tôi'}
        # Cần map đúng key từ graph_manager trả về
        src = r.get('source', '')
        tgt = r.get('target', '')
        
        # Mapping aligned with get_relations_data response
        call_listener = r.get('s_call_t', '') # Source gọi Target là gì
        call_speaker = r.get('t_call_s', 'Tôi') # (Tạm dùng logic ngược hoặc default)
        
        # New: Use explicit call_self derived from logic
        self_speaker = r.get('s_call_self', '')
        self_listener = r.get('t_call_self', '')
        
        row = f"| {src} | {tgt} | {call_listener} | {call_speaker} | {self_speaker} | {self_listener} |"
        table.append(row)
        
    return "\n".join(table)

TYPE_DISPLAY_ORDER = ["DEITY", "ORG", "LOCATION", "CONCEPT", "ITEM", "TITLE", "UNKNOWN"]
TYPE_DISPLAY_LABEL = {
    "DEITY": "Thần linh",
    "ORG": "Tổ chức",
    "LOCATION": "Địa danh",
    "CONCEPT": "Khái niệm / Pathway / Nghi lễ",
    "ITEM": "Vật phẩm",
    "TITLE": "Danh xưng / Tiếng lóng",
    "UNKNOWN": "Khác",
}


def _normalize_glossary(glossary):
    """Coerce glossary entries to the typed schema for prompt rendering.

    Accepts both the new `{term: {translation, type}}` form and the legacy
    `{term: "translation"}` form so callers do not have to migrate in lockstep.
    """
    if not glossary:
        return {}
    out = {}
    for term, value in glossary.items():
        if isinstance(value, dict):
            translation = value.get("translation", "")
            type_ = (value.get("type") or "UNKNOWN").upper()
        else:
            translation = str(value)
            type_ = "UNKNOWN"
        if not translation:
            continue
        out[term] = {"translation": translation, "type": type_}
    return out


def format_glossary_by_type(glossary):
    """Render a glossary as Markdown sections bucketed by entity type.

    A1 hierarchical retrieval — surfacing the type lets the LLM treat e.g. an ORG
    differently from a LOCATION when picking xưng hô and Hán-Việt registers.
    Returns an empty string when there are no entries to render.
    """
    normalized = _normalize_glossary(glossary)
    if not normalized:
        return ""
    buckets = {}
    for term, entry in normalized.items():
        buckets.setdefault(entry["type"], []).append((term, entry["translation"]))
    sections = []
    extras = [t for t in buckets if t not in TYPE_DISPLAY_ORDER]
    for type_ in TYPE_DISPLAY_ORDER + sorted(extras):
        items = buckets.get(type_)
        if not items:
            continue
        items.sort(key=lambda x: x[0].lower())
        label = TYPE_DISPLAY_LABEL.get(type_, type_.title())
        body = "\n".join(f"- {en} → {vi}" for en, vi in items)
        sections.append(f"## {label} ({type_})\n{body}")
    return "\n\n".join(sections)


def build_translation_prompt(chunk_text, glossary=None, relations_data=None, current_summary=None, model_type="strong"):
    """
    Xây dựng prompt dạng XML.
    """
    # 1. Chuẩn hóa dữ liệu
    glossary = glossary if glossary is not None else {}
    relations_data = relations_data if relations_data is not None else []
    current_summary = current_summary if current_summary else "Không có tóm tắt."
    
    # Check RAG mode
    is_rag_mode = bool(glossary or relations_data or current_summary != "Không có tóm tắt.")

    if is_rag_mode:
        # 2. Xử lý thành phần XML con
        
        # A. GLOSSARY SECTION (A1: hierarchical, bucketed by entity type)
        glossary_str = format_glossary_by_type(glossary)

        # B. RELATIONS SECTION (Dạng Bảng)
        relations_table = _format_relations_to_table(relations_data)

        # C. MODEL SPECIFIC INSTRUCTIONS
        if model_type == "weak":
            style_instruction = """
            <critical_rules>
                1. OUTPUT FORMAT: JSON only (theo mẫu). KHÔNG giải thích thêm.
                2. LANGUAGE: Dịch sang Tiếng Việt 100%.
                3. ENTITY TAGS: Giữ nguyên tag [Tên] sau đại từ (VD: Hắn [Klein]).
                4. GLOSSARY: Tuân thủ tuyệt đối thẻ <glossary>.
            </critical_rules>
            """
        else: # Strong models
            style_instruction = """
            <style_guide>
                - Văn phong: Tiểu thuyết phương Tây, bí ẩn, trôi chảy.
                - QUAN TRỌNG VỀ FORMAT: 
                    1. Xử lý "Pronoun [Name]":
                        + Input: "He [Zhou Mingrui]..."
                        + Output: "Đại từ ..." (Ví dụ: "Anh...").
                        + CẤM: Không được viết thiếu đại từ (VD: "[Zhou Mingrui] nhìn..." là SAI).
                    
                    2. Xử lý Tên Riêng (Name nằm ngoài tag):
                        + Tuyệt đối KHÔNG thay thế Tên Riêng bằng Đại từ.
                        + Ví dụ: Input "Zhou Mingrui sighed" -> Phải dịch "Zhou Mingrui thở dài" (KHÔNG được dịch là "Anh thở dài").
                - Sáng tạo: Có thể điều chỉnh cấu trúc câu cho tự nhiên nhưng KHÔNG thay đổi thuật ngữ trong <glossary>.
            </style_guide>
            """

        # 3. Lắp ráp Prompt XML
        
        # Prompt nền tảng
        system_prompt = f"""
    <role_definition>
    Bạn là Dịch Giả Văn Học chuyên nghiệp (Thể loại: Fantasy/Mystery).
    Nhiệm vụ: Dịch văn bản trong thẻ <input_text> sang Tiếng Việt.
    </role_definition>

    <context_data>
        <summary>
        {current_summary}
        </summary>

        <glossary>
        {glossary_str}
        </glossary>

        <relationship_matrix>
        {relations_table}
        </relationship_matrix>
    </context_data>

    {style_instruction}

    <task_requirements>
    Hãy thực hiện 2 bước suy luận (Chain of Thought):
    1. PHÂN TÍCH:
    - Xác định ai đang nói/hành động dựa trên tag [Tên].
    - Tra cứu bảng <relationship_matrix> để chọn từ xưng hô.
    - Tra cứu <glossary> để chọn thuật ngữ.
    2. BẢN DỊCH:
    - Viết bản dịch hoàn chỉnh.
    - Đại từ nhân xưng phải nhất quán với Phân Tích.
    - Giữ nguyên tên riêng
    </task_requirements>
    """

        # Input User
        user_content = f"""
    <input_text>
    {chunk_text}
    </input_text>
    """

        # 4. Few-shot Examples (Tối giản lại theo XML)
        shot_1_user = """<input_text>Dunn pointed at the bottle. "This is the Potion you need." he [Dunn] said.</input_text>"""
        shot_1_assistant = """### PHÂN TÍCH
    - Nhân vật: Dunn (Nam) đang nói.
    - Quan hệ: Dunn là cấp trên -> Xưng "Tôi/Anh".
    - Thuật ngữ: Potion -> Ma dược.

    ### BẢN DỊCH
    Dunn chỉ tay vào chiếc lọ. "Đây là lọ Ma dược cậu cần." anh nói."""

    else:
        # =========================================================================
        # MODE 2: NON-RAG TRANSLATION (XML STYLE)
        # =========================================================================
        
        # Prompt tối giản hơn vì không có dữ liệu tra cứu
        system_prompt = f"""
<role_definition>
Bạn là Dịch Giả Văn Học chuyên nghiệp (Fantasy/Mystery).
Nhiệm vụ: Dịch văn bản trong thẻ <input_text> sang Tiếng Việt.
</role_definition>

{style_instruction}

<task_requirements>
Thực hiện dịch thuật theo quy trình:
1. PHÂN TÍCH:
   - Dựa vào tag [Tên] (nếu có) để xác định nhân vật.
   - Tự suy luận xưng hô dựa trên ngữ cảnh văn bản (VD: Kẻ thù -> Ngươi/Ta; Bạn bè -> Cậu/Tớ).
2. DỊCH THUẬT:
   - Bản dịch trôi chảy, văn phong bí ẩn.
   - Tuyệt đối giữ nguyên tag [Tên] trong kết quả (VD: "Hắn [Klein] nói...").
</task_requirements>
"""

        # Few-shot cho Non-RAG (Đơn giản, tự suy luận)
        shot_1_user = """<input_text>The man sneered. "You will die here!" he [Lanevus] shouted.</input_text>"""
        
        shot_1_assistant = """### PHÂN TÍCH
- Nhân vật: Lanevus (Kẻ địch) đang đe dọa.
- Xưng hô suy luận: Hắn (Kẻ xấu) - Ngươi (Đối thủ).

### BẢN DỊCH
Gã đàn ông cười khẩy. "Ngươi sẽ phải bỏ mạng tại đây!" hắn [Lanevus] hét lên."""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": shot_1_user},
        {"role": "assistant", "content": shot_1_assistant},
        {"role": "user", "content": user_content}
    ]

    return messages

def build_literary_judge_prompt(source_text, candidate_text, reference_text=None):
    """Reference-based sentence-level LLM-judge for EN→VI literary MT (Fantasy/Mystery).

    Returns messages for an LLM call. The judge must return strict JSON:
        {"scores": {axis: 1-5, ...}, "rationale": "<1-2 sentences>"}

    Axes (all 1-5, integer):
      - accuracy: meaning faithfulness to source
      - fluency:  natural Vietnamese, novel-grade
      - register: pronouns/honorifics fit relationship & status
      - cultural: genre-specific terms in idiomatic Hán-Việt
      - style:    preserves Fantasy/Mystery narrative voice

    If reference_text is provided, the judge uses it as a gold anchor (not a target
    to match verbatim — different phrasing can still score 5).
    """
    has_ref = bool(reference_text and reference_text.strip())

    system_prompt = """Bạn là Giám khảo Dịch thuật Văn học chuyên về tiểu thuyết Fantasy/Mystery (Lord of the Mysteries, Wuxia/Xianxia, web novel).
Nhiệm vụ: Chấm bản dịch tiếng Việt theo 5 tiêu chí, mỗi tiêu chí thang 1-5 (số nguyên).

### 5 TIÊU CHÍ:

1. accuracy (Độ chính xác về nghĩa) — Bản dịch có giữ được ý nghĩa, sự kiện, sắc thái câu gốc không?
   1: Sai lệch nghĩa, bịa thông tin, mất chi tiết quan trọng.
   3: Giữ ý chính nhưng thiếu sắc thái hoặc có sai sót nhỏ.
   5: Chuyển tải hoàn chỉnh cả nghĩa và sắc thái.

2. fluency (Độ trôi chảy) — Tiếng Việt tự nhiên, đúng văn phong tiểu thuyết?
   1: Word-by-word, gượng, sai ngữ pháp.
   3: Đúng ngữ pháp nhưng cứng, lộ máy dịch.
   5: Như nhà văn chuyên nghiệp, nhịp điệu mượt.

3. register (Xưng hô / Vai vế) — Đại từ và cách xưng hô phù hợp mối quan hệ và bối cảnh?
   1: Sai xưng hô nghiêm trọng (vd cấp dưới gọi cấp trên là "mày").
   3: Trung tính, không sai nhưng thiếu cá tính.
   5: Phản ánh đúng địa vị xã hội và sắc thái nhân vật.

4. cultural (Thuật ngữ / Hán-Việt) — Thuật ngữ thể loại (tu chân, ma thuật, Tarot, vật phẩm huyền bí) dịch đúng và có sắc thái?
   1: Dịch nghĩa đen làm mất hồn ("Major Arcana" → "Bài Lớn").
   3: Đúng nghĩa nhưng dùng từ trung tính.
   5: Hán-Việt chuẩn, cổ kính (vd "Ẩn Chính", "Ma dược", "Vật Phong Ấn").

5. style (Văn phong / Giọng kể) — Giữ được giọng kể tiểu thuyết (bí ẩn, trang trọng, drama)?
   1: Phẳng, mất giọng kể.
   3: Trung tính, nhận ra thể loại.
   5: Văn phong sống động, đúng tinh thần Fantasy/Mystery.

### OUTPUT JSON STRICT (không markdown fence, không lời dẫn):
{"scores":{"accuracy":<1-5>,"fluency":<1-5>,"register":<1-5>,"cultural":<1-5>,"style":<1-5>},"rationale":"<1-2 câu nêu điểm mạnh/yếu chính>"}"""

    if has_ref:
        user_content = f"""### Source (English):
{source_text}

### Reference (Vietnamese gold — dùng làm chuẩn so sánh, KHÔNG yêu cầu giống hệt):
{reference_text}

### Candidate (Vietnamese to evaluate):
{candidate_text}

Chấm Candidate. Trả về JSON đúng format."""
    else:
        user_content = f"""### Source (English):
{source_text}

### Candidate (Vietnamese to evaluate):
{candidate_text}

Chấm Candidate (không có Reference, dựa vào Source và kiến thức Fantasy/Mystery). Trả về JSON đúng format."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_literary_pairwise_judge_prompt(source_text, candidate_a, candidate_b, reference_text=None):
    """Pairwise LLM-judge for EN→VI literary MT, framed as a Vietnamese reader's preference.

    Designed to match how a typical Vietnamese reader actually evaluates literary
    translations: prose flow + xưng hô (register), with severe word-level errors
    (vulgar mistranslation, EN/VN code-switching, Vietnamese spelling errors) as
    veto signals. Per-axis sum-of-5 absolute scoring under-weights register/style
    and over-weights surface accuracy.

    Returns messages for an LLM call. Judge must return strict JSON:
        {"choice": "A"|"B"|"tie", "rationale": "<1-2 sentences>"}

    Counter position-bias at the caller: invoke twice with A↔B swapped; only count
    as decisive if both calls agree.
    """
    has_ref = bool(reference_text and reference_text.strip())

    system_prompt = """Bạn là độc giả tiếng Việt khó tính, thường đọc tiểu thuyết Fantasy/Mystery (Lord of the Mysteries, tiên hiệp, kỳ huyễn). Nhiệm vụ: đọc 2 bản dịch tiếng Việt của cùng một câu nguồn tiếng Anh, chọn bản nào CẢM GIÁC ĐỌC tự nhiên + hấp dẫn hơn với người Việt.

### Tiêu chí ưu tiên (theo thứ tự):

1. **Văn phong & nhịp đọc** — Câu có chảy mượt như tiểu thuyết Việt không? Hay khô cứng, "máy dịch", bị động ngược, dùng từ Hán-Việt sai chỗ?

2. **Xưng hô (pronoun/register)** — Đại từ và cách gọi có khớp địa vị nhân vật và bối cảnh không? Sai "ông/anh/hắn/y" là lỗi nặng với độc giả Việt, dù nội dung đúng.

3. **Lỗi nghiêm trọng (red flag — auto pick bản còn lại)**:
   - Từ tục tĩu / sai chính tả tiếng Việt (vd "Lồn" thay vì "Loen", "absurđ" thay vì "absurd/vô lý")
   - Code-switching tiếng Anh giữa câu tiếng Việt
   - Bịa thông tin / mất chi tiết then chốt làm hiểu sai cốt truyện

4. **Độ chính xác về nghĩa** — Phụ trợ. Bản dịch khác nhau về cách diễn đạt nhưng cùng giữ ý chính thì KHÔNG là yếu tố quyết định.

### Quy tắc chọn:

- Chọn "A" nếu A đọc hay/tự nhiên hơn rõ rệt theo tiêu chí 1-2, hoặc B có lỗi red flag.
- Chọn "B" nếu B đọc hay/tự nhiên hơn rõ rệt, hoặc A có lỗi red flag.
- Chọn "tie" CHỈ khi cả hai bản cho cảm giác đọc tương đương — không có khác biệt rõ về văn phong, không bản nào sai xưng hô nặng hơn, không bản nào có red flag.
- KHÔNG chọn "tie" để né quyết định: nếu nghiêng dù chỉ một chút về cảm giác đọc, phải chọn A hoặc B.

### Output JSON STRICT (không markdown fence, không lời dẫn):
{"choice":"A"|"B"|"tie","rationale":"<1-2 câu nêu lý do — focus vào văn phong/xưng hô/red flag, KHÔNG liệt kê 5 trục>"}"""

    if has_ref:
        user_content = f"""### Source (English):
{source_text}

### Reference (Vietnamese gold — bản dịch của người dịch chuyên nghiệp, dùng để tham khảo văn phong/xưng hô CHUẨN, KHÔNG yêu cầu A hoặc B giống hệt):
{reference_text}

### Candidate A:
{candidate_a}

### Candidate B:
{candidate_b}

Hãy đọc cả 2 và chọn. Trả về JSON đúng format."""
    else:
        user_content = f"""### Source (English):
{source_text}

### Candidate A:
{candidate_a}

### Candidate B:
{candidate_b}

Hãy đọc cả 2 và chọn (không có Reference, dựa vào kiến thức Fantasy/Mystery và cảm giác đọc tiếng Việt). Trả về JSON đúng format."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_evaluation_prompt(source_text, translation_a, translation_b):
    """
    Update: Chuyển sang cơ chế chấm điểm (1-10) để tìm Winner thay vì chỉ định tính.
    """
    
    system_instruction = """Bạn là Giám khảo Văn học chuyên nghiệp. Nhiệm vụ của bạn là chấm điểm và so sánh hai bản dịch (Translation A và Translation B) từ văn bản gốc.

### TIÊU CHÍ CHẤM ĐIỂM (Thang 1-10):
1. **Accuracy (Độ chính xác)**: Ý nghĩa bảo toàn bao nhiêu % so với gốc? 
   - 1: Sai lệch hoàn toàn, bịa đặt thông tin.
   - 10: Chính xác tuyệt đối, giữ được cả sắc thái nghĩa.
2. **Fluency (Độ trôi chảy)**: Văn phong có tự nhiên như người Việt viết không? 
   - 1: Như Google Translate cũ, sượng, sai ngữ pháp.
   - 10: Như nhà văn chuyên nghiệp, văn phong tiểu thuyết lôi cuốn.
3. **Terminology Logic (Logic thuật ngữ)**: Trong phạm vi đoạn văn này, thuật ngữ dùng có hợp lý và nhất quán không?

### QUY TẮC OUTPUT (JSON ONLY):
Bạn phải trả về kết quả dưới dạng JSON (không Markdown, không lời dẫn), theo cấu trúc sau:

```json
{
    "model_a": {
        "accuracy_score": 8,
        "fluency_score": 7,
        "terminology_score": 8,
        "comment": "Nhận xét ngắn về A..."
    },
    "model_b": {
        "accuracy_score": 9,
        "fluency_score": 9,
        "terminology_score": 9,
        "comment": "Nhận xét ngắn về B..."
    },
    "winner": "B",
    "reasoning": "Giải thích lý do tại sao B thắng (dựa trên tổng điểm và trải nghiệm đọc)."
}
"""
    real_user_content = f"""Source text: {source_text}

Translation A: {translation_a}

Translation B: {translation_b}

Instruction: Compare Translation A and Translation B. Decide which is better based ONLY on the criteria above. Return the result in strict JSON format. """
    
    messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": real_user_content}
]

    return messages


def format_tm_examples(examples):
    """Format Translation Memory retrieval results as few-shot block.

    Args:
        examples: list of {source, translation, similarity, score?} from
                  TranslationMemory.retrieve(); pass empty/None to skip.
    """
    if not examples:
        return ""
    lines = [
        "<translation_memory>",
        "Các bản dịch chất lượng cao có nội dung tương tự (tham khảo văn phong, cách chọn từ, tên riêng):",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\n[Ví dụ {i}] (similarity={ex['similarity']:.2f})")
        lines.append(f"EN: {ex['source']}")
        lines.append(f"VI: {ex['translation']}")
    lines.append("</translation_memory>")
    return "\n".join(lines)


# --- Helper: Build Draft Translation Prompt (Phase 1 - Drafting) ---
def build_draft_translation_prompt(chunk_text, prev_context="", tm_examples=None):
    """
    Build prompt for Phase 1 (Drafting) - Pure literary translation.
    No entity tags, no glossary constraints. Focus on fluency.

    Args:
        chunk_text: The source text to translate
        prev_context: Optional context from previous chunk (last 2 sentences)
        tm_examples: Optional list of Translation Memory retrieval results
                     (from TranslationMemory.retrieve) to inject as few-shot.

    Returns:
        list: Messages for LLM call
    """
    system_prompt = """
<role_definition>
Bạn là Tiểu thuyết gia chuyên nghiệp, chuyên dịch tiểu thuyết Fantasy/Mystery.
Nhiệm vụ: Dịch văn bản sang Tiếng Việt với văn phong trôi chảy, tự nhiên nhất.
</role_definition>

<style_guide>
- Văn phong: Tiểu thuyết phương Tây, bí ẩn, lôi cuốn
- Ưu tiên: Độ mượt mà, tự nhiên của câu văn
- KHÔNG cần tuân theo bất kỳ quy tắc xưng hô hay thuật ngữ cố định nào
- Dịch thoát ý, tránh dịch word-by-word
- Nếu có <translation_memory>: học văn phong và cách dùng từ từ các ví dụ đó,
  KHÔNG copy nguyên văn, KHÔNG dịch lại các câu trong ví dụ.
</style_guide>

<output_format>
Chỉ trả về bản dịch, không cần phân tích hay giải thích.
</output_format>
"""

    context_section = f"\n<previous_context>\n{prev_context}\n</previous_context>\n" if prev_context else ""
    tm_section = format_tm_examples(tm_examples)
    tm_block = f"\n{tm_section}\n" if tm_section else ""

    user_content = f"""
{context_section}{tm_block}
<input_text>
{chunk_text}
</input_text>
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


# --- Helper: Build Post-Editing Prompt (Phase 2 - Editing) ---
def build_post_editing_prompt(source_text, draft_text, injected_reference, glossary, relations):
    """
    Build prompt for Phase 2 (Post-Editing) - Refine draft with entity context.
    
    Args:
        source_text: Original source text (unused, kept for compatibility)
        draft_text: Draft translation from Phase 1
        injected_reference: Text with [Entity] annotations for reference
        glossary: Dict of {English term: Vietnamese translation}
        relations: List of relation dicts with pronoun rules
        
    Returns:
        list: Messages for LLM call
    """
    # Format glossary as type-grouped sections (A1: hierarchical retrieval)
    glossary_str = format_glossary_by_type(glossary) or "Không có thuật ngữ đặc biệt."
    
    # Format relations as table (same as _format_relations_to_table)
    relations_table = _format_relations_to_table(relations)
    
    # Check if source has entity tags
    has_entity_tags = "[" in injected_reference and "]" in injected_reference
    has_relations = bool(relations_table)
    
    # Use injected_reference as the source (it contains [Entity] tags)
    # Fallback to original source_text if no tags present
    if has_entity_tags:
        annotated_source = injected_reference
    else:
        annotated_source = source_text
    
    # Build constraints based on what's available
    # Tag rules: only if has_entity_tags
    # Relationship reference: only if has_relations
    
    if has_entity_tags:
        if has_relations:
            # Has tags AND relations: full explanation
            tag_rules = """
1. HIỂU VỀ TAG [Tên]: Trong <source_text>, các đại từ (he, she, him, her...) được gắn tag [Tên nhân vật] để chỉ rõ đại từ đó đang nói về AI.
   - Ví dụ: "He [Klein] said..." nghĩa là "He" ở đây chính là nhân vật Klein.
   - Dùng thông tin này để xác định đúng đại từ xưng hô tiếng Việt theo <relationship_matrix>.

2. CÁCH SỬ DỤNG TAG:
   - Tra cứu tên trong tag [Tên] với <relationship_matrix> để biết nhân vật đó tự xưng là gì, gọi người khác là gì.
   - Áp dụng vào bản dịch: Thay "anh ấy/cô ấy" bằng đại từ phù hợp (Tôi, Hắn, Ngài...).
"""
            fix_rules = """
4. CHỈ SỬA các lỗi về:
   - Thuật ngữ (theo <glossary>)
   - Đại từ xưng hô (theo <relationship_matrix> và tag [Tên])
"""
        else:
            # Has tags but NO relations: explain tags but don't mention relationship_matrix
            tag_rules = """
1. HIỂU VỀ TAG [Tên]: Trong <source_text>, các đại từ (he, she, him, her...) được gắn tag [Tên nhân vật] để chỉ rõ đại từ đó đang nói về AI.
   - Ví dụ: "He [Klein] said..." nghĩa là "He" ở đây chính là nhân vật Klein.
   - Dùng tag này để xác định ai đang nói/hành động, nhưng KHÔNG có thông tin xưng hô cụ thể.
"""
            fix_rules = """
3. CHỈ SỬA các lỗi về thuật ngữ (theo <glossary>).
"""
    else:
        # No tags: simple mode
        tag_rules = ""
        fix_rules = """
2. CHỈ SỬA các lỗi về thuật ngữ (theo <glossary>).
"""
    
    # Build relationship_matrix section only if has_relations
    if has_relations:
        relationship_section = f"""
<relationship_matrix>
{relations_table}
</relationship_matrix>"""
    else:
        relationship_section = ""
    
    # Assemble constraints
    base_rules = """
3. GIỮ NGUYÊN văn phong và cách hành văn của bản nháp khi nó đã đúng.
""" if has_entity_tags else """
1. GIỮ NGUYÊN văn phong và cách hành văn của bản nháp khi nó đã đúng.
"""
    
    no_hallucination = """
5. KHÔNG thêm thông tin mới, KHÔNG bịa đặt nội dung.
""" if has_entity_tags and has_relations else """
4. KHÔNG thêm thông tin mới, KHÔNG bịa đặt nội dung.
""" if has_entity_tags else """
3. KHÔNG thêm thông tin mới, KHÔNG bịa đặt nội dung.
"""
    
    constraints = f"""
<critical_constraints>
{tag_rules}{base_rules}{fix_rules}{no_hallucination}</critical_constraints>"""
    
    system_prompt = f"""
<role_definition>
Bạn là Biên tập viên Văn học chuyên nghiệp.
Nhiệm vụ: Chỉnh sửa bản dịch nháp để đảm bảo tính nhất quán về thuật ngữ và đại từ xưng hô.
</role_definition>

{constraints}

<glossary>
{glossary_str}
</glossary>
{relationship_section}
"""

    user_content = f"""
<source_text>
{annotated_source}
</source_text>

<draft_translation>
{draft_text}
</draft_translation>

Hãy chỉnh sửa <draft_translation> dựa trên các quy tắc trên. Chỉ trả về bản dịch đã chỉnh sửa.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]