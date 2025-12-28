import json

# --- Helper: Build prompt ---
def build_extraction_prompt(chunk_text, existing_nodes_list, spacy_candidates_list):
    """
    Update: Tăng cường ràng buộc Tiếng Việt và Hán Việt.
    """
    system_instruction = """Bạn là một Chuyên gia Kỹ thuật Knowledge Graph cho hệ thống dịch thuật tiểu thuyết.
Nhiệm vụ: Trích xuất Thực thể (Nodes) và Mối quan hệ (Edges) quan trọng từ văn bản.

### 1. QUY TẮC BỘ LỌC (QUAN TRỌNG NHẤT - TRÁNH RÁC):
Chúng ta chỉ quan tâm đến các thực thể có vai trò cốt truyện, **LOẠI BỎ HOÀN TOÀN** các đồ vật làm nền.

* **Banned Entities (CẤM LẤY):**
    * **Đồ nội thất/Vật dụng:** Bàn (desk), ghế (chair), giường (bed), tủ (cabinet), đèn (lamp), cửa sổ (window)... -> **KHÔNG PHẢI LÀ LOCATION.**
    * **Vật phẩm phổ thông:** Bút (pen), mực (ink), giấy (paper), súng thường (gun), tiền xu (coin), sách vở (books/notebook)... -> **TRỪ KHI** nó là vật phẩm ma thuật có TÊN RIÊNG (Ví dụ: "Antigonus Notebook", "0-08"). Nếu chỉ là "cuốn sổ", "khẩu súng" -> **BỎ QUA.**
    * **Bộ phận cơ thể:** Tay, chân, mắt, đầu... -> **BỎ QUA.**

* **Valid Types (CHỈ CHẤP NHẬN CÁC LOẠI SAU):**
    1.  `PERSON`: Nhân vật có tên riêng hoặc danh xưng cụ thể (VD: Klein, Captain, The Fool).
    2.  `LOCATION`: Địa danh địa lý, Tòa nhà, Thành phố, Quốc gia (VD: Tingen, Blackthorn Security Company). **Lưu ý:** Một căn phòng chỉ được tính là LOCATION nếu nó có tên riêng quan trọng (VD: "Phòng Tiên Tri"), còn "phòng ngủ", "phòng khách" chung chung thì BỎ QUA.
    3.  `ORGANIZATION`: Tổ chức, Giáo hội, Biệt đội (VD: Nighthawks, Church of the Evernight Goddess).
    4.  `SPECIAL_OBJECT`: Chỉ lấy vật phẩm ma thuật, các vị thần huyển bí có **Tên Riêng** (Proper Noun).

### 2. QUY TẮC NGÔN NGỮ OUTPUT (BẮT BUỘC TUÂN THỦ):
* `reasoning`: PHẢI viết bằng **TIẾNG VIỆT**.
* `label`: Dịch tên nhân vật/địa danh sang **Hán Việt** hoặc **Tiếng Việt** chuẩn.
    * VD: "Klein" -> "Klein"
    * "Zhou Mingrui" -> "Zhou Mingrui"
    * "Evernight Goddess" -> "Nữ Thần Đêm Đen"

### 3. NODE RESOLUTION:
* Ưu tiên dùng ID từ `EXISTING_NODES` nếu trùng khớp.
* Chỉ chọn các thực thể trong danh sách `CANDIDATES`. Tuy nhiên, nếu candidate là một từ khóa bị cấm (như "cái bàn", "khẩu súng"), hãy **LỌC BỎ** nó ngay lập tức.

### 4. QUY TẮC CHỌN SOURCE VÀ TARGET:
* **CHỈ CHỌN TỪ CANDIDATES**: Bạn chỉ được trích xuất các thự thể trong danh sách `CANDIDATES` được cung cấp ở đầu vào.

### 5. QUY TẮC XƯNG HÔ (PRONOUNS) - CỰC KỲ QUAN TRỌNG:
* **Dựa trên Hội thoại**: Nếu có hội thoại, hãy trích xuất cách xưng hô thực tế.
* **Dựa trên Vị thế (Implicit Status)**: Nếu KHÔNG có hội thoại, hãy suy luận từ vai trò:
    * **DEITY / LEADER vs MEMBER**: 
        * Member -> Leader: "Ngài" (Lord/Sir), "Đội trưởng" (Captain).
        * Leader -> Member: "Ngươi" (You), "Cậu/Cô" (You).
    * **STRANGER vs STRANGER**: "Anh/Hắn" - "Tôi".
    * **ENEMY**: "Hắn/Gã/Y" - "Ta/Ngươi".
* **Lưu ý đặc biệt**: Với thực thể là "The Fool" (Kẻ Khờ) hoặc các vị thần, Target (kẻ dưới) LUÔN LUÔN gọi là "Ngài" (My Lord), bất kể quan hệ là ALLY.

### OUTPUT SCHEMA (Strict JSON):
```json
{
  "reasoning": "1. Loại bỏ 'Study Desk', 'Ink Bottle' vì là đồ vật thông thường. 2. Trích xuất 'Zhou Mingrui' vì là nhân vật chính...",
  "nodes": [
    {
      "id": "Standard_English_ID",
      "label": "label", 
      "type": "PERSON" | "LOCATION" | "ORGANIZATION" | "SPECIAL_OBJECT",
      "gender": "MALE" | "FEMALE" | "UNKNOWN"
    }
  ],
  "edges": [
    {
      "source": "ID_A",
      "target": "ID_B",
      "relation_type": "ALLY" | "ENEMY" | "NEUTRAL" | "MASTER_DISCIPLE" | "FAMILY" | "ROMANTIC",
      "interaction_summary": "Tóm tắt tương tác ngắn gọn (Tiếng Việt)",
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
    
    
    shot_1_assistant = """```json { "reasoning": "1. 'Klein' khớp với node cũ 'Klein Moretti'. 2. 'Dunn Smith' là nhân vật mới, vai trò đội trưởng (cấp trên). 3. Bối cảnh: Cấp trên khiển trách cấp dưới. 4. Xưng hô: Dunn gọi Klein là 'Cậu', Klein gọi Dunn là 'Đội trưởng'.", "nodes": [ { "id": "Klein Moretti", "label": "Klein Moretti", "type": "PERSON", "gender": "MALE" }, { "id": "Dunn Smith", "label": "Dunn Smith", "type": "PERSON", "gender": "MALE" } ], "edges": [ { "source": "Dunn Smith", "target": "Klein Moretti", "relation_type": "ALLY", "interaction_summary": "Dunn Smith khiển trách Klein vì đến muộn.", "pronouns": { "source_call_target": "Cậu", "target_call_source": "Đội trưởng" } } ] }"""

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
    """
    Update: Thêm category CONCEPT/TITLE để bắt các thuật ngữ trừu tượng (Pathways, Potions, Tarot terms).
    """
    system_instruction = """
Bạn là một Biên tập viên Từ điển Cấp cao cho dự án Dịch thuật Tiểu thuyết Giả tưởng (theo phong cách Quỷ Bí Chi Chủ / Lord of the Mysteries).
Nhiệm vụ của bạn là trích xuất các thuật ngữ cụ thể và cung cấp bản dịch tiếng Việt chuẩn xác nhất cho chúng.

### 1. CÁC DANH MỤC MỤC TIÊU (Cần trích xuất):
* **TỔ CHỨC (ORGANIZATION)**: Các Giáo phái, Giáo hội, Biệt đội, Công ty, Hiệp hội Kinh tế. (Ví dụ: Kẻ Gác Đêm, Hiệp hội Than Thép).
* **THẦN LINH (DEITY)**: Các vị Thần, Thực thể thần thoại. (Ví dụ: Nữ Thần Đêm Đen).
* **ĐỊA DANH (LOCATION)**: Vương quốc, Thành phố, Tên đường, Các tòa nhà có tên riêng.
* **KHÁI NIỆM / DANH XƯNG / VẬT PHẨM**: 
    * **Đường tắt/Danh sách (Pathways/Sequences)**: (Ví dụ: Nhà Tiên Tri, Khán Giả, Người Phi Phàm).
    * **Thuật ngữ Ma thuật**: (Ví dụ: Ma dược, Nghi thức, Minh tưởng).
    * **Vật phẩm cụ thể**: (Ví dụ: Bài Tarot, Vật Phong Ấn, 0-08).
    * **Danh xưng/Tiếng lóng**: (Ví dụ: Anh hùng bàn phím, Ẩn Chính).

### 2. QUY TẮC LOẠI TRỪ (Bỏ qua những từ này):
* **BỎ QUA DANH TỪ CHUNG**: Bàn, ghế, đèn, súng thường, tiền, bánh mì (trừ khi đó là một vật phẩm có tên riêng cụ thể).
* **BỎ QUA ĐỘNG TỪ/TÍNH TỪ**: Trừ khi chúng là một phần của danh từ riêng hoặc thuật ngữ chuyên môn.

### 3. HƯỚNG DẪN DỊCH THUẬT:
* **Dịch theo ngữ cảnh**: KHÔNG dịch từng từ (word-for-word). Hãy sử dụng kiến thức chuyên ngành (domain knowledge).
    * *Ví dụ*: "Major Arcana" -> "Ẩn Chính" (Thuật ngữ Tarot), KHÔNG dịch là "Đại Arcana".
    * *Ví dụ*: "Amalgamators" (trong kinh doanh) -> "Hiệp hội/Tổ hợp", KHÔNG dịch là "Hợp kim".
* **Văn phong**: Ưu tiên sử dụng từ **Hán Việt** cho các thuật ngữ Ma thuật/Tu chân/Tôn giáo để tạo sắc thái trang trọng, cổ kính.

### ĐỊNH DẠNG ĐẦU RA (CHỈ JSON):
```json
{
  "reasoning": "Giải thích lý do tại sao bạn trích xuất các thuật ngữ này và cách bạn chọn từ ngữ dịch thuật...",
  "glossary": {
      "English Term": "Vietnamese Translation"
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
    "reasoning": "1. 'Major Arcana' is a Tarot Concept -> CONCEPT. Translate: 'Ẩn Chính'. 2. 'Keyboard Warrior' is a Slang/Title -> CONCEPT. Translate: 'Anh hùng bàn phím' or 'Thánh bàn phím'.", 
    "glossary": { "Major Arcana": "Ẩn Chính", "Keyboard Warrior": "Anh hùng bàn phím" 
    } 
}"""

    # Few-shot 2: Filtering noise (Person & Common items)
    shot_2_user = """
    TEXT:
    Klein sat on the sofa in the living room. He looked at Old Neil. The Holy Cathedral was far away.
    """
    
    shot_2_assistant = """```json
{
  "reasoning": "1. 'Klein' and 'Old Neil' are PERSONS -> IGNORE. 2. 'sofa', 'living room' are generic objects/places -> IGNORE. 3. 'Holy Cathedral' is a named building/organization seat -> LOCATION/ORG. Translate: 'Thánh Đường'.",
  "glossary": {
      "Holy Cathedral": "Thánh Đường"
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

import json

def build_translation_prompt(chunk_text, glossary=None, relations_str=None, current_summary=None, model_type="strong"):
    """
    Xây dựng prompt dịch thuật hỗ trợ 2 chế độ (RAG/Non-RAG) và 2 loại Model (Strong/Weak).
    
    Args:
        model_type (str): "strong" (Gemini Pro/GPT-4) hoặc "weak" (Qwen 7B/14B, Llama).
    """
    
    # Chuẩn hóa dữ liệu đầu vào
    glossary = glossary if glossary is not None else {}
    relations_str = relations_str if relations_str is not None else ""
    current_summary = current_summary if current_summary is not None else ""

    # Xác định chế độ: Nếu có dữ liệu hỗ trợ thì là RAG
    is_rag_mode = bool(glossary or relations_str or current_summary)

    # --- XÂY DỰNG INSTRUCTION ADD-ON DỰA TRÊN MODEL TYPE ---
    if model_type == "weak": 
        # Cho 7B, 14B (Qwen, Llama) - Cần chỉ dẫn cực kỳ nghiêm ngặt để tránh lỗi format
        instruction_add_on = """
### CẢNH BÁO QUAN TRỌNG (CRITICAL):
1. OUTPUT BẮT BUỘC LÀ TIẾNG VIỆT 100%. CẤM dùng Tiếng Trung hay Tiếng Anh giải thích.
2. TUÂN THỦ TUYỆT ĐỐI bảng Glossary. Không sáng tạo lại các thuật ngữ đã cung cấp.
3. Giữ nguyên định dạng JSON/Output như yêu cầu. Không thêm lời dẫn."""
    else: 
        # Cho Gemini Flash/Pro, GPT-4 - Cho phép sáng tạo hơn
        instruction_add_on = """
### HƯỚNG DẪN VĂN PHONG:
Bạn có quyền linh hoạt điều chỉnh câu từ để bản dịch tự nhiên nhất, miễn là không làm sai lệch ý nghĩa cốt lõi và các thuật ngữ trong Glossary."""

    if is_rag_mode:
        # =========================================================================
        # MODE 1: RAG TRANSLATION
        # =========================================================================
        
        matched_glossary = {}
        for key, value in glossary.items():
            if key in chunk_text:
                matched_glossary[key] = value

        # Sử dụng f-string để chèn instruction_add_on
        system_instruction = f"""Bạn là một Dịch Giả Văn Học chuyên nghiệp, chuyên biên dịch tiểu thuyết Kỳ ảo phương Tây (Western Fantasy) mang màu sắc trinh thám và huyền bí.
Nhiệm vụ của bạn là dịch đoạn văn bản tiếng Anh được cung cấp sang Tiếng Việt.

### 1. QUY TẮC BẮT BUỘC (STRICT RULES):
* **OUTPUT**: Bản dịch BẮT BUỘC là Tiếng Việt
* **TUYỆT ĐỐI TUÂN THỦ GLOSSARY**: Nếu từ khóa tiếng Anh nằm trong 'BẢNG THUẬT NGỮ', bạn PHẢI dùng đúng từ tiếng Việt tương ứng.
* **XƯNG HÔ (PRONOUNS)**: 
    - **Ưu tiên 1 (Dữ liệu đầu vào)**: Kiểm tra kỹ phần `QUAN HỆ & XƯNG HÔ`. Nếu có chỉ định, BẮT BUỘC phải tuân theo.
    - **Ưu tiên 2 (Tự suy luận - Fallback)**: Nếu phần quan hệ bị TRỐNG hoặc thiếu nhân vật, bạn PHẢI tự xác định dựa trên ngữ cảnh văn bản (xét theo Địa vị, Tuổi tác, Độ thân thiết, Phe phái).
    - **Tính Nhất Quán (Consistency)**: Dù lấy từ dữ liệu hay tự suy luận, cặp xưng hô đã chọn phải được **giữ nguyên xuyên suốt** toàn bộ bản dịch, không được thay đổi giữa chừng.
* **VĂN PHONG**: 
    - Giữ nguyên Tiếng Anh tên nhân vật, KHÔNG DỊCH ra Tiếng Việt.
    - Dùng từ Hán Việt cho các thuật ngữ tu luyện (Ví dụ: Ma dược, Phi phàm giả).
    - Văn phong trôi chảy, thoát ý, giữ giọng văn bí ẩn.

{instruction_add_on}

### 2. QUY TRÌNH THỰC HIỆN (CHAIN OF THOUGHT):
Bạn phải thực hiện theo 2 bước trong câu trả lời:
1. **### PHÂN TÍCH**: (BẮT BUỘC VIẾT BẰNG TIẾNG VIỆT). Tóm tắt ngữ cảnh, xác định cặp nhân vật...
2. **### BẢN DỊCH**: Bản dịch hoàn chỉnh.
"""

        # --- FEW-SHOT 1 (RAG) ---
        shot_1_input = """
--- DỮ LIỆU ĐẦU VÀO ---
### 1. TÓM TẮT CHƯƠNG (CONTEXT)
Dunn Smith đang giải thích cho Klein về công việc.
### 2. BẢNG THUẬT NGỮ (GLOSSARY)
{ "Nighthawks": "Kẻ Gác Đêm", "Potion": "Ma dược" }
### 3. QUAN HỆ & XƯNG HÔ (RELATIONS)
Dunn Smith (Captain) -> Klein Moretti (Member) [ALLY]
### 4. VĂN BẢN CẦN DỊCH (INPUT TEXT)
Dunn pointed at the bottle. "This is the Potion you need. Welcome to the Nighthawks," he said calmly.
"""
        shot_1_output = """### PHÂN TÍCH
- Ngữ cảnh: Dunn hướng dẫn Klein nhận ma dược gia nhập đội.
- Xưng hô: Dunn (Anh/Đội trưởng) - Klein (Tôi/Cậu). Dunn gọi Klein là "Cậu".
- Thuật ngữ áp dụng: Potion -> Ma dược, Nighthawks -> Kẻ Gác Đêm.

### BẢN DỊCH
Dunn chỉ tay vào chiếc lọ. "Đây là lọ Ma dược cậu cần. Chào mừng gia nhập Kẻ Gác Đêm," anh bình thản nói."""

        # --- FEW-SHOT 2 (RAG) ---
        shot_2_input = """
--- DỮ LIỆU ĐẦU VÀO ---
### 1. TÓM TẮT CHƯƠNG (CONTEXT)
Klein đối mặt với một tên tội phạm điên cuồng.
### 2. BẢNG THUẬT NGỮ (GLOSSARY)
{}
### 3. QUAN HỆ & XƯNG HÔ (RELATIONS)
Lanevus (Enemy) <-> Klein Moretti (Enemy) [HOSTILE]
### 4. VĂN BẢN CẦN DỊCH (INPUT TEXT)
The man sneered, his eyes filled with madness. "You think you can stop me? You will die here!"
"""
        shot_2_output = """### PHÂN TÍCH
- Ngữ cảnh: Tên tội phạm đe dọa Klein trong trận chiến.
- Xưng hô: Hắn (Kẻ địch) - Klein (Ngươi/Mày). Hắn xưng "Ta", gọi Klein là "Ngươi".
- Thuật ngữ áp dụng: Không có.

### BẢN DỊCH
Gã đàn ông cười khẩy, ánh mắt ngập tràn điên loạn. "Ngươi nghĩ mình có thể ngăn cản ta sao? Ngươi sẽ phải bỏ mạng tại đây!\""""

        # --- REAL INPUT (RAG) ---
        real_user_input = f"""
--- DỮ LIỆU ĐẦU VÀO ---

### 1. TÓM TẮT CHƯƠNG (CONTEXT)
{current_summary}

### 2. BẢNG THUẬT NGỮ (GLOSSARY - BẮT BUỘC DÙNG)
{json.dumps(matched_glossary, ensure_ascii=False)}

### 3. QUAN HỆ & XƯNG HÔ (RELATIONS)
{relations_str}

### 4. VĂN BẢN CẦN DỊCH (INPUT TEXT)
{chunk_text}
"""
    else:
        # =========================================================================
        # MODE 2: NON-RAG TRANSLATION
        # =========================================================================
        
        system_instruction = f"""Bạn là một Dịch Giả Văn Học chuyên nghiệp, chuyên biên dịch tiểu thuyết Kỳ ảo phương Tây (Western Fantasy) mang màu sắc trinh thám và huyền bí.
Nhiệm vụ của bạn là dịch đoạn văn bản tiếng Anh được cung cấp sang Tiếng Việt.

### 1. QUY TẮC DỊCH THUẬT:
* **VĂN PHONG**: 
    - Giữ nguyên Tiếng Anh tên nhân vật, KHÔNG DỊCH ra Tiếng Việt.
    - Dùng từ Hán Việt cho các thuật ngữ tu luyện (Ví dụ: Ma dược, Phi phàm giả).
    - Văn phong trôi chảy, thoát ý, giữ giọng văn bí ẩn.
* **XƯNG HÔ**: Tự suy luận dựa trên ngữ cảnh văn bản (Ví dụ: Bạn bè dùng Cậu-Tớ, Kẻ thù dùng Ngươi-Ta, Quý tộc dùng Ngài).

{instruction_add_on}

### 2. QUY TRÌNH THỰC HIỆN (CHAIN OF THOUGHT):
Bạn phải thực hiện theo 2 bước trong câu trả lời:
1. **### PHÂN TÍCH**: (BẮT BUỘC VIẾT BẰNG TIẾNG VIỆT). Tóm tắt ngữ cảnh, xác định cặp nhân vật...
2. **### BẢN DỊCH**: Bản dịch hoàn chỉnh.
"""

        # --- FEW-SHOT 1 (NON-RAG) ---
        shot_1_input = """Dunn pointed at the bottle. "This is the Potion you need. Welcome to the Nighthawks," he said calmly."""
        
        shot_1_output = """### PHÂN TÍCH
- Ngữ cảnh: Nhân vật Dunn đang hướng dẫn người khác, không khí bình tĩnh.
- Xưng hô dự đoán: Cấp trên nói với cấp dưới hoặc người hướng dẫn (Anh - Cậu/Tôi).
- Từ vựng: Potion -> Ma dược, Nighthawks -> Kẻ Gác Đêm.

### BẢN DỊCH
Dunn chỉ tay vào chiếc lọ. "Đây là lọ Ma dược cậu cần. Chào mừng gia nhập Kẻ Gác Đêm," anh bình thản nói."""

        # --- FEW-SHOT 2 (NON-RAG) ---
        shot_2_input = """The man sneered, his eyes filled with madness. "You think you can stop me? You will die here!\""""
        
        shot_2_output = """### PHÂN TÍCH
- Ngữ cảnh: Xung đột, đe dọa, kẻ nói đang điên loạn.
- Xưng hô dự đoán: Kẻ địch (Ta - Ngươi/Mày).
- Giọng văn: Gắt gỏng, điên cuồng.

### BẢN DỊCH
Gã đàn ông cười khẩy, ánh mắt ngập tràn điên loạn. "Ngươi nghĩ mình có thể ngăn cản ta sao? Ngươi sẽ phải bỏ mạng tại đây!\""""

        # --- REAL INPUT (NON-RAG) ---
        real_user_input = chunk_text

    # --- RETURN MESSAGES (Common for both) ---
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": shot_1_input},
        {"role": "assistant", "content": shot_1_output},
        {"role": "user", "content": shot_2_input},
        {"role": "assistant", "content": shot_2_output},
        {"role": "user", "content": real_user_input}
    ]
    
    return messages

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