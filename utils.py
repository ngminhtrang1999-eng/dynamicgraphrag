import re 
import json
import json_repair
import os
import torch
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

# --- Helper: Create chunk with overlap (prevent context loss) ---
def split_text_with_overlap(text, max_words=500, overlap_ratio=0.15):
    """
    Chia text thành các chunk dựa trên số từ, đảm bảo trọn vẹn câu (dùng Spacy).
    Các chunk sau sẽ gối đầu (overlap) lên chunk trước 1 khoảng bằng overlap_ratio.
    """
    # Tăng giới hạn ký tự cho spacy đề phòng chương quá dài
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    
    # Lấy danh sách các câu (Span objects)
    sentences = list(doc.sents)
    n_sents = len(sentences)
    
    chunks = []
    start_idx = 0  # Index của câu bắt đầu chunk hiện tại
    
    while start_idx < n_sents:
        current_word_count = 0
        end_idx = start_idx
        
        # 1. Forward: Gom câu cho đến khi đủ max_words
        while end_idx < n_sents:
            # Lấy số token (tương đương số từ)
            sent_len = len(sentences[end_idx])
            
            # Nếu thêm câu này mà vượt quá max_words (và chunk không rỗng) thì dừng
            if current_word_count + sent_len > max_words and current_word_count > 0:
                break
            
            current_word_count += sent_len
            end_idx += 1
        
        # Tạo text cho chunk hiện tại
        # Ghép các câu từ start_idx đến end_idx
        chunk_text = "".join([sent.text_with_ws for sent in sentences[start_idx:end_idx]])
        chunks.append(chunk_text)
        
        # Nếu đã đi đến hết văn bản thì dừng
        if end_idx >= n_sents:
            break
            
        # 2. Backward: Tính toán điểm bắt đầu cho chunk sau (để lấy overlap)
        overlap_target = int(current_word_count * overlap_ratio)
        overlap_accumulated = 0
        new_start_idx = end_idx
        
        # Lùi lại từ cuối chunk hiện tại để lấy đủ lượng từ overlap
        while new_start_idx > start_idx:
            prev_sent_len = len(sentences[new_start_idx - 1])
            if overlap_accumulated + prev_sent_len > overlap_target:
                # Đã đủ overlap, dừng lùi
                break
            overlap_accumulated += prev_sent_len
            new_start_idx -= 1
            
        # Safety check: Đảm bảo luôn tiến lên ít nhất 1 câu để tránh vòng lặp vô tận
        # (Trường hợp 1 câu quá dài chiếm hết cả chunk và overlap)
        if new_start_idx <= start_idx:
            new_start_idx = start_idx + 1
            
        start_idx = new_start_idx

    return chunks


# --- Helper: Clean NER ---
def get_spacy_candidates(text):
    nlp.max_length = len(text) + 100000
    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        # print(ent.text, ent.label_)
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT"]:
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