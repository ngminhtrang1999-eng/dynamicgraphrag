import re 
import json
import json_repair
import os
import torch
import difflib
import semchunk
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import const
import spacy

try:
    nlp = spacy.load("en_core_web_trf")
except:
    print("⚠️ Warning: en_core_web_trf not found, falling back to sm.")
    nlp = spacy.load("en_core_web_sm")

# --- Helper: JSON IO ---
def load_json(path, default):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    return default

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- Helper: Robust JSON Parser ---
def extract_json_from_response(response_text):
    # Dùng thư viện json_repair để tự động tìm và sửa lỗi JSON
    try:
        decoded_object = json_repair.loads(response_text)
        return decoded_object
    except Exception as e:
        print("--- Raw Response ---")
        print(response_text) 
        print(f"❌ Vẫn không thể parse JSON: {e}")
        return None

def extract_api_response_content(response_json):
    """
    Extracts content from OpenRouter or LangDB API response (OpenAI-compatible format).
    """
    try:
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            print(f"⚠️ Unexpected API response format: {response_json}")
            return None
    except Exception as e:
        print(f"❌ Error extracting API content: {e}")
        return None

# --- Helper: Clear duplicate candidates ---
def filter_similar_names(candidates):
    # Bước 1: Loại bỏ trùng lặp chính xác và sắp xếp theo độ dài giảm dần
    # Lý do: Chúng ta muốn ưu tiên giữ lại cái tên dài nhất (Klein Moretti) trước
    sorted_candidates = sorted(list(set(candidates)), key=len, reverse=True)
    
    final_list = []
    
    # Danh sách các từ danh xưng cần loại bỏ khi so sánh
    stop_words = {'mr', 'mr.', 'ms', 'ms.', 'mrs', 'mrs.', 'dr', 'dr.'}

    for cand in sorted_candidates:
        is_subset = False
        
        # Chuẩn bị tập hợp từ (tokens) cho cái tên đang xét
        # Chuyển về chữ thường để so sánh không phân biệt hoa thường
        cand_tokens = set(cand.lower().split())
        
        # Lọc bỏ các từ stop_words (ví dụ: 'mr.' trong 'mr. klein' sẽ bị bỏ, chỉ còn 'klein')
        cleaned_cand_tokens = {t for t in cand_tokens if t not in stop_words}

        for existing in final_list:
            existing_lower = existing.lower()
            existing_tokens = set(existing_lower.split())

            # Kịch bản 1: Chuỗi con (Substring)
            # Ví dụ: "Klein" nằm trong "Klein Moretti"
            if cand.lower() in existing_lower:
                is_subset = True
                break
            
            # Kịch bản 2: Tập hợp con (Token Subset)
            # Ví dụ: "Mr. Klein" -> {'klein'} là tập con của "Klein Moretti" -> {'klein', 'moretti'}
            if cleaned_cand_tokens and cleaned_cand_tokens.issubset(existing_tokens):
                is_subset = True
                break
        
        # Nếu không phải là tập con của bất kỳ tên nào đã chọn, thêm vào danh sách
        if not is_subset:
            final_list.append(cand)
            
    return final_list


# --- Helper: Check Chinese in text ---
def has_chinese_char(text):
    # Regex kiểm tra dải ký tự CJK (Chinese-Japanese-Korean) thông dụng
    pattern = re.compile(r'[\u4e00-\u9fff]')
    if pattern.search(text):
        return True
    return False


# --- Helper: Clean Entity Tags from Translation ---
def clean_entity_tags(text):
    """
    Remove all [Name] entity tags and fix resulting double spaces.
    """
    # 1. Xóa tất cả các tag [Name]
    # Ta không cần \s* ở đây, cứ xóa tag trước đã
    text_no_tags = re.sub(r'\[[^\]]+\]', '', text)
    
    # 2. Thay thế tất cả 'nhiều dấu cách' (double spaces) thành '1 dấu cách'
    # Pattern \s+ nghĩa là tìm 1 hoặc nhiều khoảng trắng liền nhau
    cleaned = re.sub(r' +', ' ', text_no_tags)
    
    # 3. (Tuỳ chọn) Xóa khoảng trắng thừa trước dấu chấm/phẩy (nếu có)
    # Ví dụ: "He turned ." -> "He turned."
    cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)

    return cleaned.strip()


# # --- Helper: Align and Fix Tags ---
# def align_and_fix_tags(original_text, annotated_text):
#     """
#     Recover tags from annotated_text while strictly enforcing original_text content.
#     Algorithm: 
#     1. Extract tags and map them to the index of the character they follow in the clean text.
#     2. Use SequenceMatcher to find matching blocks between Clean Annotated vs Original.
#     3. Reconstruct Original, inserting tags at matching indices.
#     """
    
#     # --- Bước 1: Tách Tags và tạo Clean Text + Tag Map ---
#     # Tag Map: {index_trong_clean_text: [List các tag cần chèn sau ký tự này]}
    
#     clean_chars = []
#     tag_map = {} 
    
#     # Tách chuỗi dựa trên pattern tag [Name]
#     # Ví dụ: "Hắn [Klein]" -> ['Hắn ', '[Klein]', '']
#     parts = re.split(r'(\[[^\]]+\])', annotated_text)
    
#     # current_clean_idx là index của ký tự cuối cùng vừa được thêm vào clean_chars
#     current_clean_idx = -1 
    
#     # Xử lý trường hợp Tag nằm ngay đầu câu (hiếm gặp nhưng cần handle)
#     start_tags = []
    
#     for part in parts:
#         if not part: continue
        
#         if part.startswith('[') and part.endswith(']'):
#             # Đây là Tag
#             if current_clean_idx == -1:
#                 start_tags.append(part)
#             else:
#                 if current_clean_idx not in tag_map:
#                     tag_map[current_clean_idx] = []
#                 tag_map[current_clean_idx].append(part)
#         else:
#             # Đây là Text
#             for char in part:
#                 clean_chars.append(char)
#                 current_clean_idx += 1
                
#     clean_text = "".join(clean_chars)
    
#     # --- Bước 2: Khớp Clean Text với Original Text ---
#     # autojunk=False giúp chính xác hơn với text dài
#     matcher = difflib.SequenceMatcher(None, clean_text, original_text, autojunk=False)
    
#     result_chars = []
    
#     # Chèn các tag ở ngay đầu văn bản (nếu có)
#     if start_tags:
#         result_chars.extend(start_tags)
#         result_chars.append(" ") # Thêm space an toàn
        
#     last_orig_pos = 0
    
#     # --- Bước 3: Duyệt qua các khối văn bản giống nhau ---
#     # get_matching_blocks trả về các block (a, b, size) khớp nhau
#     # a: start index bên clean_text
#     # b: start index bên original_text
#     for block in matcher.get_matching_blocks():
#         a_start, b_start, size = block
        
#         # 3a. Lấp đầy khoảng trống (Gap) từ Original Text
#         # Nếu có đoạn text trong Original không khớp với Clean (do LLM xóa/sửa), 
#         # ta phải chèn lại nguyên vẹn từ Original để giữ format.
#         if b_start > last_orig_pos:
#             result_chars.append(original_text[last_orig_pos:b_start])
            
#         # 3b. Xử lý đoạn khớp nhau từng ký tự một
#         for i in range(size):
#             # Index hiện tại trong Clean Text và Original Text
#             clean_idx = a_start + i
#             orig_idx = b_start + i
            
#             char = original_text[orig_idx]
#             result_chars.append(char)
            
#             # Kiểm tra xem tại clean_idx này có tag nào cần chèn không
#             if clean_idx in tag_map:
#                 tags = tag_map[clean_idx]
#                 for tag in tags:
#                     # Logic thêm dấu cách thông minh:
#                     # Nếu ký tự trước đó không phải khoảng trắng, thêm cách.
#                     # (Tránh dính chữ: "him[Tag]" -> "him [Tag]")
#                     if char.strip(): 
#                         result_chars.append(" ")
                    
#                     result_chars.append(tag)

#                     if orig_idx + 1 < len(original_text):
#                         next_char = original_text[orig_idx + 1]
#                         # Nếu next_char là chữ cái hoặc số, thêm space để tách ra
#                         if next_char.isalnum():
#                             result_chars.append(" ")
        
#         last_orig_pos = b_start + size
        
#     # --- Bước 4: Chèn nốt phần đuôi của Original Text (nếu còn) ---
#     if last_orig_pos < len(original_text):
#         result_chars.append(original_text[last_orig_pos:])
        
#     return "".join(result_chars)

# --- Helper: Semantic Chunking (No Overlap, Sentence Preserving) ---
def count_tokens(text_or_messages):
    """
    Count tokens using tiktoken (cl100k_base).
    Supports string or list of messages (simple concat estimation for messages).
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback if encoding not found, though cl100k_base is standard
        return len(str(text_or_messages).split())

    if isinstance(text_or_messages, str):
        return len(encoder.encode(text_or_messages))
    elif isinstance(text_or_messages, list):
        # Rough estimation for messages list
        content = ""
        for msg in text_or_messages:
            content += str(msg.get("content", "")) + " "
        return len(encoder.encode(content))
    return 0

def count_sentences(text):
    """Count sentences using Spacy."""
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    return len(list(doc.sents))

def split_text_with_overlap(text, max_words=500, overlap_ratio=0.15):
    """
    Split text using semantic chunking (semchunk) based on token count.
    
    Args:
        text: Input text
        max_words: Target chunk size (in TOKENS, despite parameter name)
        overlap_ratio: IGNORED (kept for compatibility)
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        token_counter = lambda x: len(encoder.encode(x))
    except Exception:
        print("⚠️ tiktoken 'cl100k_base' load failed. Using split() fallback.")
        token_counter = lambda x: len(x.split())

    # semchunk handles the splitting respecting sentence boundaries
    chunks = semchunk.chunk(text, chunk_size=max_words, token_counter=token_counter)
    
    return chunks



# --- Helper: Clean NER ---
def get_spacy_candidates(text):
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        # print(ent.text, ent.label_)
        # if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"]:
        if ent.label_ in ["PERSON"]:
            clean = re.sub(r"['’]s\b", "", ent.text, flags=re.IGNORECASE)
            clean = re.sub(r"[^\w\s]", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > 2 and not clean.isdigit():
                names.add(clean)
    return list(names)


# --- Helper: Preprocess data for evaluation ---
def preprocess_evaluate_data(raw_data_list):
    """
    Hàm gom nhóm lại dữ liệu dựa trên Reference.
    Nếu Ref bị trống hoặc là <MERGED_ABOVE>, nó sẽ nối Src và MT vào câu trước đó.
    """
    cleaned_data = []
    
    for item in raw_data_list:
        src = str(item['src']) if item['src'] else ""
        mt = str(item['mt']) if item['mt'] else ""
        ref = str(item['ref']) if item['ref'] else ""
        
        # Logic: Nếu Ref có dữ liệu và KHÔNG PHẢI merged -> Bắt đầu một mẫu đánh giá mới
        if ref and ref.strip() != "" and ref != "<MERGED_ABOVE>":
            cleaned_data.append({
                "src": src,
                "mt": mt, # MT có thể là <MERGED_ABOVE>, sẽ xử lý sau hoặc chấp nhận text rác? 
                          # Tốt nhất là clean ngay: nếu mt <MERGED_ABOVE> thì coi như chưa có nội dung, 
                          # NHƯNG thường dòng đầu block không bao giờ là MERGED_ABOVE cho target của nó.
                "ref": ref
            })
        else:
            # Logic: Nếu Ref trống -> Đây là phần đuôi của câu trước
            # Ta nối Src và MT của câu này vào Src và MT của câu trước đó
            if len(cleaned_data) > 0:
                last_item = cleaned_data[-1]
                
                # Nối Src (luôn nối để giữ context nếu cần, nhưng BLEURT ko dùng src)
                # last_item['src'] += " " + src 
                
                # Nối MT: CHỈ NỐI NẾU KHÔNG PHẢI LÀ <MERGED_ABOVE>
                if mt != "<MERGED_ABOVE>" and mt.strip() != "":
                    last_item['mt'] += " " + mt
                    
                # Ref giữ nguyên vì nó đã bao hàm cả ý này rồi
            else:
                # Trường hợp dòng đầu tiên đã bị lỗi Ref
                pass

    # Clean up: Remove <MERGED_ABOVE> from start if any (hiếm gặp)
    final_data = []
    for item in cleaned_data:
        if item['mt'] == "<MERGED_ABOVE>":
            item['mt'] = "" # Reset nếu bị lỗi
        final_data.append(item)
        
    return final_data

# --- Helper: Infer pronoun pair ---
def infer_pronoun_pair(call_term, gender="UNKNOWN"):
    """
    Suy luận từ "Xưng" dựa trên từ "Gọi" (Hô) và Giới tính.
    Input: "Cháu", gender="MALE" -> Output: "Chú"
    """
    call_term = call_term.lower().strip()
    gender = gender.upper().strip() if gender else "UNKNOWN"
    
    # Logic Map thủ công
    # 1. Logic phụ thuộc giới tính (Gender-dependent)
    if call_term in ["cháu", "con"]:
        if gender == "MALE": return "Chú" # Hoặc Bác/Ông tùy context, nhưng Chú là an toàn
        if gender == "FEMALE": return "Cô" # Hoặc Dì/Bà
        return "Chú/Cô"
        
    if call_term in ["em", "nhóc", "bé"]:
        if gender == "MALE": return "Anh"
        if gender == "FEMALE": return "Chị"
        return "Anh/Chị"
        
    # 2. Logic cơ bản (Gender-independent or default)
    mapping = {
        # Kính ngữ / Cấp trên
        "đội trưởng": "Tôi",
        "ngài": "Tôi",
        "thầy": "Con",
        "cô": "Tôi",
        
        # Ngang hàng / Xã giao
        "anh": "Tôi",
        "cậu": "Tôi", 
        "cô ấy": "Tôi",
        "quý ngài": "Tôi",
        "bà": "Tôi",
        "ông": "Tôi",
        
        # Gia đình (Hard to exact without full context, simplify)
        "anh": "Em",
        "chị": "Em",
        
        # Thù địch / Suồng sã
        "ngươi": "Ta",
        "mày": "Tao",
        "hắn": "Ta",
        "gã": "Ta",
        "nó": "Ta"
    }
    
    # Check mapping
    if call_term in mapping:
        return mapping[call_term]
        
    # Fallback
    return "Tôi"


# --- Helper: Add placeholder for pronouns ---
def add_placeholder_regex(text):
    # Danh sách các pronouns bạn muốn nhắm tới
    # (?i) là cờ để không phân biệt hoa thường (He = he)
    # \b là ranh giới từ (đảm bảo "he" đứng riêng lẻ, không nằm trong "the")
    pattern = r"(?i)\b(he|she|him|her|his)\b"
    
    # Thay thế từ tìm được bằng: Chính nó (\g<0>) + dấu cách + []
    modified_text = re.sub(pattern, r"\g<0> []", text)
    
    return modified_text

# --- Helper: Get Last Sentences ---
def get_last_sentences(text, n=2):
    """
    Get the last n sentences from the text using Spacy.
    """
    if not text:
        return ""
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    if len(sentences) <= n:
        return " ".join(sentences)
    
    return " ".join(sentences[-n:])
